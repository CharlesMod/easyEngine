// Package gguf reads the header, metadata and tensor table of a GGUF file (v2/v3). It never
// reads tensor data, so a 10 GB model is described from the first few hundred KiB.
package gguf

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"sort"
	"strings"
)

// Layer describes one transformer block's KV cache cost.
type Layer struct {
	HasKV     bool // false for recurrent/linear-attention blocks (no KV cache)
	Recurrent bool // linear-attention/SSM block: a fixed state per slot instead of a KV cache
	SWA       bool // sliding-window attention: cache capped at the window per slot
	KVHeads   int
	HeadDimK  int
	HeadDimV  int
}

// Info is what the planner needs from a model file. The scalar KV fields describe the
// full-attention layers (the first non-SWA layer; layer 0 when there is no SWA); models that
// mix layer kinds (Gemma 4) also fill Layers, which the planner prefers when present.
type Info struct {
	Arch          string
	NLayer        int
	NKVHead       int
	HeadDimK      int
	HeadDimV      int
	SlidingWindow int
	SWALayers     int // number of sliding-window layers
	NExpert       int
	NExpertUsed   int
	ExpertBytes   int64 // tensors named *_exps*
	DenseBytes    int64 // everything else
	FileSize      int64

	ExpertLayers int     // layers that hold *_exps* tensors
	Layers       []Layer // per layer, len == NLayer when known
	HeadDimKSWA  int
	HeadDimVSWA  int
	NTensors     int

	// KVLoraRank > 0 marks MLA attention: the cache holds the compressed latent plus the rope
	// part per token (HeadDimK = kv_lora_rank + rope dims, one head) and no separate V (HeadDimV 0).
	KVLoraRank int
	// StateBytesPerSlot is the recurrent state (f32) of every linear-attention/SSM layer for one
	// slot. It does not grow with context.
	StateBytesPerSlot int64
	// Estimated is set when the header lacked what the KV cost needs (unknown architecture):
	// the numbers are a conservative guess (every layer full multi-head attention), and Note says
	// what was assumed.
	Estimated bool
	Note      string
}

// Tensor is one tensor-table entry.
type Tensor struct {
	Name   string
	Dims   []uint64
	Type   uint32
	Offset uint64
	Bytes  int64
}

const ggufMagic = 0x46554747 // "GGUF" little endian

type kind uint32

const (
	kU8 kind = iota
	kI8
	kU16
	kI16
	kU32
	kI32
	kF32
	kBool
	kStr
	kArr
	kU64
	kI64
	kF64
)

var kindSize = map[kind]int{kU8: 1, kI8: 1, kU16: 2, kI16: 2, kU32: 4, kI32: 4, kF32: 4, kBool: 1, kU64: 8, kI64: 8, kF64: 8}

// ggml type id -> {block elements, bytes per block}
var typeSize = map[uint32][2]int64{
	0: {1, 4}, 1: {1, 2}, 2: {32, 18}, 3: {32, 20}, 6: {32, 22}, 7: {32, 24}, 8: {32, 34}, 9: {32, 36},
	10: {256, 84}, 11: {256, 110}, 12: {256, 144}, 13: {256, 176}, 14: {256, 210}, 15: {256, 292},
	16: {256, 66}, 17: {256, 74}, 18: {256, 98}, 19: {256, 50}, 20: {32, 18}, 21: {256, 110},
	22: {256, 82}, 23: {256, 136}, 24: {1, 1}, 25: {1, 2}, 26: {1, 4}, 27: {1, 8}, 28: {1, 8},
	29: {256, 56}, 30: {1, 2}, 34: {256, 54}, 35: {256, 66}, 39: {32, 17},
}

type reader struct {
	r   *bufio.Reader
	err error
}

func (rd *reader) bytes(n int) []byte {
	if rd.err != nil {
		return nil
	}
	b := make([]byte, n)
	if _, err := io.ReadFull(rd.r, b); err != nil {
		rd.err = err
		return nil
	}
	return b
}
func (rd *reader) u32() uint32 {
	b := rd.bytes(4)
	if b == nil {
		return 0
	}
	return binary.LittleEndian.Uint32(b)
}
func (rd *reader) u64() uint64 {
	b := rd.bytes(8)
	if b == nil {
		return 0
	}
	return binary.LittleEndian.Uint64(b)
}
func (rd *reader) discard(n int64) {
	if rd.err != nil {
		return
	}
	for n > 0 {
		step := n
		if step > math.MaxInt32 {
			step = math.MaxInt32
		}
		d, err := rd.r.Discard(int(step))
		n -= int64(d)
		if err != nil {
			rd.err = err
			return
		}
	}
}
func (rd *reader) str() string {
	n := rd.u64()
	if rd.err != nil {
		return ""
	}
	if n > 1<<28 {
		rd.err = fmt.Errorf("gguf: string of %d bytes is implausible", n)
		return ""
	}
	return string(rd.bytes(int(n)))
}

// value reads a metadata value. Strings are kept (short ones only matter); numeric and bool
// arrays up to 65536 elements are kept as []float64; other arrays are skipped.
func (rd *reader) value(t kind) any {
	switch t {
	case kStr:
		return rd.str()
	case kArr:
		et := kind(rd.u32())
		n := rd.u64()
		if rd.err != nil {
			return nil
		}
		if et == kStr {
			for i := uint64(0); i < n && rd.err == nil; i++ {
				l := rd.u64()
				rd.discard(int64(l))
			}
			return nil
		}
		if et == kArr {
			rd.err = errors.New("gguf: nested arrays are not supported")
			return nil
		}
		sz, ok := kindSize[et]
		if !ok {
			rd.err = fmt.Errorf("gguf: unknown array element type %d", et)
			return nil
		}
		if n > 65536 {
			rd.discard(int64(n) * int64(sz))
			return nil
		}
		out := make([]float64, 0, n)
		for i := uint64(0); i < n && rd.err == nil; i++ {
			out = append(out, rd.scalar(et))
		}
		return out
	default:
		if _, ok := kindSize[t]; !ok {
			rd.err = fmt.Errorf("gguf: unknown value type %d", t)
			return nil
		}
		return rd.scalar(t)
	}
}

func (rd *reader) scalar(t kind) float64 {
	b := rd.bytes(kindSize[t])
	if b == nil {
		return 0
	}
	switch t {
	case kU8, kBool:
		return float64(b[0])
	case kI8:
		return float64(int8(b[0]))
	case kU16:
		return float64(binary.LittleEndian.Uint16(b))
	case kI16:
		return float64(int16(binary.LittleEndian.Uint16(b)))
	case kU32:
		return float64(binary.LittleEndian.Uint32(b))
	case kI32:
		return float64(int32(binary.LittleEndian.Uint32(b)))
	case kF32:
		return float64(math.Float32frombits(binary.LittleEndian.Uint32(b)))
	case kU64:
		return float64(binary.LittleEndian.Uint64(b))
	case kI64:
		return float64(int64(binary.LittleEndian.Uint64(b)))
	case kF64:
		return math.Float64frombits(binary.LittleEndian.Uint64(b))
	}
	return 0
}

// File is the parsed header.
type File struct {
	Version  uint32
	KV       map[string]any // string, float64 (all numbers and bools), or []float64
	Tensors  []Tensor
	FileSize int64
}

// Parse reads the header, metadata and tensor table from r. fileSize is used to size the
// last tensor when its type is not in the size table.
func Parse(r io.Reader, fileSize int64) (*File, error) {
	rd := &reader{r: bufio.NewReaderSize(r, 1<<20)}
	if m := rd.u32(); rd.err != nil || m != ggufMagic {
		if rd.err != nil {
			return nil, fmt.Errorf("gguf: %w", rd.err)
		}
		return nil, errors.New("gguf: bad magic (not a GGUF file)")
	}
	f := &File{KV: map[string]any{}, FileSize: fileSize}
	f.Version = rd.u32()
	if f.Version < 2 || f.Version > 3 {
		return nil, fmt.Errorf("gguf: unsupported version %d", f.Version)
	}
	nt, nkv := rd.u64(), rd.u64()
	if rd.err != nil {
		return nil, fmt.Errorf("gguf: %w", rd.err)
	}
	if nt > 1<<20 || nkv > 1<<20 {
		return nil, fmt.Errorf("gguf: implausible counts (%d tensors, %d keys)", nt, nkv)
	}
	for i := uint64(0); i < nkv; i++ {
		k := rd.str()
		t := kind(rd.u32())
		v := rd.value(t)
		if rd.err != nil {
			return nil, fmt.Errorf("gguf: metadata %d (%q): %w", i, k, rd.err)
		}
		if v != nil {
			f.KV[k] = v
		}
	}
	for i := uint64(0); i < nt; i++ {
		var t Tensor
		t.Name = rd.str()
		nd := rd.u32()
		if nd > 8 {
			return nil, fmt.Errorf("gguf: tensor %q has %d dims", t.Name, nd)
		}
		for d := uint32(0); d < nd; d++ {
			t.Dims = append(t.Dims, rd.u64())
		}
		t.Type = rd.u32()
		t.Offset = rd.u64()
		if rd.err != nil {
			return nil, fmt.Errorf("gguf: tensor %d: %w", i, rd.err)
		}
		f.Tensors = append(f.Tensors, t)
	}
	f.sizeTensors()
	return f, nil
}

// sizeTensors fills Tensor.Bytes from the type table; unknown types take the gap to the next
// tensor offset (the last one runs to the end of the file).
func (f *File) sizeTensors() {
	idx := make([]int, len(f.Tensors))
	for i := range idx {
		idx[i] = i
	}
	sort.Slice(idx, func(a, b int) bool { return f.Tensors[idx[a]].Offset < f.Tensors[idx[b]].Offset })
	for n, i := range idx {
		t := &f.Tensors[i]
		if ts, ok := typeSize[t.Type]; ok {
			elems := int64(1)
			for _, d := range t.Dims {
				elems *= int64(d)
			}
			t.Bytes = elems / ts[0] * ts[1]
			if elems%ts[0] != 0 {
				t.Bytes += ts[1]
			}
			continue
		}
		if n+1 < len(idx) {
			t.Bytes = int64(f.Tensors[idx[n+1]].Offset - t.Offset)
		} else if f.FileSize > 0 {
			// data section start is unknown here; this overestimates by the header size
			t.Bytes = f.FileSize - int64(t.Offset)
			if t.Bytes < 0 {
				t.Bytes = 0
			}
		}
	}
}

// Read opens path and returns its Info.
func Read(path string) (Info, error) {
	fh, err := os.Open(path)
	if err != nil {
		return Info{}, err
	}
	defer fh.Close()
	st, err := fh.Stat()
	if err != nil {
		return Info{}, err
	}
	f, err := Parse(fh, st.Size())
	if err != nil {
		return Info{}, fmt.Errorf("%s: %w", path, err)
	}
	return f.Info()
}

func (f *File) num(key string) (float64, bool) {
	v, ok := f.KV[key]
	if !ok {
		return 0, false
	}
	n, ok := v.(float64)
	return n, ok
}

func (f *File) arr(key string) ([]float64, bool) {
	v, ok := f.KV[key]
	if !ok {
		return nil, false
	}
	a, ok := v.([]float64)
	return a, ok
}

// Info derives the planner's view of the model.
func (f *File) Info() (Info, error) {
	var in Info
	a, ok := f.KV["general.architecture"].(string)
	if !ok || a == "" {
		return in, errors.New("gguf: no general.architecture")
	}
	in.Arch = a
	key := func(s string) string { return a + "." + s }
	geti := func(s string) int { n, _ := f.num(key(s)); return int(n) }

	in.NLayer = geti("block_count")
	var notes []string
	if in.NLayer <= 0 {
		for _, t := range f.Tensors {
			if i, ok := blockIndex(t.Name); ok && i+1 > in.NLayer {
				in.NLayer = i + 1
			}
		}
		if in.NLayer <= 0 {
			in.NLayer = 32
			notes = append(notes, "layer count unknown, assumed 32")
		} else {
			notes = append(notes, "block_count missing, layer count taken from the tensor names")
		}
		in.Estimated = true
	}
	in.NExpert = geti("expert_count")
	in.NExpertUsed = geti("expert_used_count")
	in.SlidingWindow = geti("attention.sliding_window")
	in.NTensors = len(f.Tensors)
	in.FileSize = f.FileSize

	heads := geti("attention.head_count")
	kvHeads := make([]int, in.NLayer)
	if arr, ok := f.arr(key("attention.head_count_kv")); ok {
		for i := range kvHeads {
			if i < len(arr) {
				kvHeads[i] = int(arr[i])
			} else if len(arr) > 0 {
				kvHeads[i] = int(arr[len(arr)-1])
			}
		}
	} else {
		n := geti("attention.head_count_kv")
		if n == 0 {
			n = heads
		}
		for i := range kvHeads {
			kvHeads[i] = n
		}
	}
	if arr, ok := f.arr(key("attention.head_count")); ok && heads == 0 && len(arr) > 0 {
		heads = int(arr[0])
	}

	kvLora := geti("attention.kv_lora_rank")
	if kvLora > 0 { // MLA: one latent "head" per token
		for i := range kvHeads {
			if kvHeads[i] > 0 {
				kvHeads[i] = 1
			}
		}
		in.KVLoraRank = kvLora
	}
	recur := make([]bool, in.NLayer)
	for i := range kvHeads {
		recur[i] = kvHeads[i] == 0
	}

	// Layer kinds. Gemma 4: bool pattern (true = sliding window). Gemma 3: an integer period
	// (every Nth layer is full attention). Qwen hybrids: full_attention_interval (others are
	// recurrent and hold no KV).
	swa := make([]bool, in.NLayer)
	hasKV := make([]bool, in.NLayer)
	for i := range hasKV {
		hasKV[i] = true
	}
	if arr, ok := f.arr(key("attention.sliding_window_pattern")); ok && len(arr) > 0 {
		if len(arr) == 1 && arr[0] > 1 { // stored as a period in an array of one
			for i := range swa {
				swa[i] = (i+1)%int(arr[0]) != 0
			}
		} else {
			for i := range swa {
				swa[i] = arr[i%len(arr)] != 0
			}
		}
	} else if n, ok := f.num(key("attention.sliding_window_pattern")); ok && n > 1 {
		for i := range swa {
			swa[i] = (i+1)%int(n) != 0
		}
	} else if in.SlidingWindow > 0 {
		// a window with no pattern: every layer is windowed (Mistral-style)
		for i := range swa {
			swa[i] = true
		}
	}
	if in.SlidingWindow == 0 {
		for i := range swa {
			swa[i] = false
		}
	}
	if n, ok := f.num(key("full_attention_interval")); ok && n > 1 {
		for i := range hasKV {
			hasKV[i] = (i+1)%int(n) == 0
			recur[i] = !hasKV[i]
		}
	}

	if n := geti("attention.shared_kv_layers"); n > 0 && n < in.NLayer {
		for i := in.NLayer - n; i < in.NLayer; i++ { // these reuse an earlier layer's cache
			hasKV[i] = false
		}
	}

	emb := geti("embedding_length")
	dk, dv := geti("attention.key_length"), geti("attention.value_length")
	if dk == 0 && heads > 0 {
		dk = emb / heads
	}
	if dv == 0 {
		dv = dk
	}
	if kvLora > 0 {
		lat := kvLora + geti("rope.dimension_count")
		if dk < lat { // some converters write the uncompressed head size here
			dk = lat
		}
		dv = 0 // llama.cpp keeps no V cache for MLA
	}
	dkS, dvS := geti("attention.key_length_swa"), geti("attention.value_length_swa")
	if dkS == 0 {
		dkS = dk
	}
	if dvS == 0 {
		dvS = dv
	}
	in.HeadDimKSWA, in.HeadDimVSWA = dkS, dvS

	for i := range recur {
		if recur[i] && !sharedKV(in.NLayer, geti("attention.shared_kv_layers"), i) {
			hasKV[i] = false
		} else {
			recur[i] = false
		}
	}
	anyKV := false
	for i := range hasKV {
		if hasKV[i] && kvHeads[i] > 0 {
			anyKV = true
		}
	}
	nRecur := 0
	for i := range recur {
		if recur[i] {
			nRecur++
		}
	}
	if dk == 0 || (!anyKV && nRecur == 0) {
		// An architecture this reader does not know: assume every layer is full multi-head
		// attention at 128 dims. Over-estimates, never zero.
		if heads == 0 {
			heads = 32
		}
		if emb > 0 && emb/heads > 0 {
			dk = emb / heads
		} else {
			dk = 128
		}
		dv = dk
		for i := range kvHeads {
			kvHeads[i], hasKV[i], recur[i] = heads, true, false
			swa[i] = false
		}
		in.SWALayers, in.Estimated = 0, true
		notes = append(notes, fmt.Sprintf("attention layout of %q unknown, assumed %d full layers with %d heads of %d dims", a, in.NLayer, heads, dk))
		nRecur = 0
	}

	// Recurrent state per slot: linear attention (Kimi delta attention), SSM, or a guess.
	var perLayer int64
	if nRecur > 0 {
		conv := geti("ssm.conv_kernel")
		if conv <= 0 {
			conv = 4
		}
		switch {
		case geti("kda.head_dim") > 0 && heads > 0:
			hd := int64(geti("kda.head_dim"))
			nh := int64(heads)
			perLayer = (3*int64(conv-1)*nh*hd + hd*hd*nh) * 4
		case geti("ssm.state_size") > 0 && geti("ssm.inner_size") > 0:
			d, inner, grp := int64(geti("ssm.state_size")), int64(geti("ssm.inner_size")), int64(geti("ssm.group_count"))
			perLayer = (int64(conv-1)*(inner+2*grp*d) + d*inner) * 4
		default:
			if heads > 0 && emb > 0 {
				perLayer = int64(emb) * int64(emb) / int64(heads) * 4
			} else {
				perLayer = 16 << 20
			}
			in.Estimated = true
			notes = append(notes, "recurrent layers with an unknown state layout, state size guessed")
		}
		in.StateBytesPerSlot = perLayer * int64(nRecur)
	}

	first := 0
	for i := range swa {
		if swa[i] {
			in.SWALayers++
		}
	}
	for i := range swa {
		if !swa[i] && hasKV[i] {
			first = i
			break
		}
	}
	in.NKVHead, in.HeadDimK, in.HeadDimV = kvHeads[first], dk, dv
	in.Layers = make([]Layer, in.NLayer)
	for i := range in.Layers {
		l := Layer{HasKV: hasKV[i], SWA: swa[i], KVHeads: kvHeads[i], HeadDimK: dk, HeadDimV: dv,
			Recurrent: recur[i]}
		if swa[i] {
			l.HeadDimK, l.HeadDimV = dkS, dvS
		}
		if kvHeads[i] == 0 {
			l.HasKV = false
		}
		in.Layers[i] = l
	}

	exLayers := map[int]bool{}
	for _, t := range f.Tensors {
		if strings.Contains(t.Name, "_exps") {
			in.ExpertBytes += t.Bytes
			if i, ok := blockIndex(t.Name); ok {
				exLayers[i] = true
			}
		} else {
			in.DenseBytes += t.Bytes
		}
	}
	in.ExpertLayers = len(exLayers)
	in.Note = strings.Join(notes, "; ")
	return in, nil
}

func blockIndex(name string) (int, bool) {
	rest, ok := strings.CutPrefix(name, "blk.")
	if !ok {
		return 0, false
	}
	n := 0
	digits := 0
	for _, c := range rest {
		if c < '0' || c > '9' {
			break
		}
		n = n*10 + int(c-'0')
		digits++
	}
	return n, digits > 0
}

// sharedKV reports whether layer i reuses an earlier layer's cache (the last n layers).
func sharedKV(nLayer, n, i int) bool { return n > 0 && n < nLayer && i >= nLayer-n }
