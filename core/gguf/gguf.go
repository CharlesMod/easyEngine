// Package gguf reads the header, metadata and tensor table of a GGUF file (v2/v3). It never
// reads tensor data, so a 10 GB model is described from the first few hundred KiB.
//
// One walk of the header makes one reading (ReadHeader, File): every metadata key with an integer,
// bool, string or short integer array value, the tensor table where asked, and the chat
// template; Info is derived from that reading. A header it cannot read whole is a typed error,
// never a guess, a value cut to fit, or an allocation of what the header merely claims.
//
// It is part of the sizing core (docs/ENGINE.md section 13): it reads from an io.Reader the
// caller opened and a size the caller gives, and must never open a file itself, import os, or
// import a package of the engine's root module. The engine's plan.ReadInfo opens a path and
// calls Parse.
package gguf

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
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
	DenseBytes    int64 // everything else (includes HostBytes)
	// HostBytes is the part of DenseBytes llama.cpp keeps in system RAM even with -ngl 999: the
	// per-layer token embeddings of the Gemma E-series (a lookup table, never a matmul).
	HostBytes int64
	// VocabSize is the token count, from token_embd.weight; it sets the size of the output
	// (logits) buffers on the card.
	VocabSize int
	FileSize  int64
	// CtxTrain is the context the model was trained for (<arch>.context_length), in tokens;
	// 0 when the header does not say it (never a guess).
	CtxTrain int

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
	// BlockBytes is the bytes of the tensors of each repeating block (named blk.N.*), len ==
	// NLayer: what moving block N off the card moves (llama.cpp -ngl moves whole blocks). A tensor
	// that llama.cpp places with a block but names without one (Gemma 4's per_layer_model_proj, kept
	// with block 0) is not counted in any. One entry per layer (zeros
	// where no tensor names the block); nil where NLayer is 0. Since the dense split (in no core tag yet).
	BlockBytes []int64
	// OutputBytes is the output head that goes to the card with any block (llama.cpp's output
	// layer): output.weight and output_norm.weight, or where the file has no output.weight
	// (OutputTied: the head is token_embd.weight read a second time) token_embd.weight and the norm.
	OutputBytes int64
	OutputTied  bool
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

// Bounds of the reading. A header past them is refused (ErrPastSize), never cut to fit.
const (
	// MaxString is the longest string the reader takes, in bytes (a chat template is a few KiB).
	MaxString = 1 << 28
	// MaxIntArray is the most elements an array of integers or bools may have and be kept: enough
	// for any per-layer figure. A longer one (a vocabulary's token types) is walked past.
	MaxIntArray = 1 << 16
	maxCount    = 1 << 20 // keys, and tensors
	maxRank     = 8
	maxLayers   = 1 << 16 // block_count, and the block index in a tensor name
)

// What a header may claim, beyond the bounds above (each refusal is ErrMalformed or ErrPastSize):
//
//   - keys and tensors: at most 1<<20 each, and the table must fit the size given;
//   - tensor rank: at most 8 dimensions (the format's own limit is 4), each dimension at least 1
//     and at most MaxInt64; the element count and the byte size (elements / block elements * bytes
//     per block, rounded up) must not pass MaxInt64, and nor may the sums of tensor bytes (dense,
//     expert) or, for ParamCount, of element counts; a tensor offset must not pass MaxInt64;
//   - <arch>.block_count, where present: 0 to 65536 layers (no real model has more than a few
//     hundred); where absent, or 0 (read as not said, as before), the count is taken from the
//     tensor names under the same bound;
//   - the block index in a tensor name (blk.N.): at most 65535; the digits are read with the
//     overflow checked, never wrapped. An index at or past a stated block_count is kept (a file may
//     carry extra blocks, as nextn/MTP layers): nothing is sized from it then;
//   - strings at most MaxString bytes and arrays at most what the size given can hold; integer
//     arrays are kept only to MaxIntArray elements.

// The reader's refusals. Each is wrapped with where it was met (a key or a tensor by its index);
// errors.Is tells them apart.
var (
	// ErrMagic: the stream does not begin with the GGUF magic.
	ErrMagic = errors.New("gguf: bad magic (not a GGUF file)")
	// ErrCut: the stream ended inside the header.
	ErrCut = errors.New("gguf: the header is cut short")
	// ErrPastSize: a length or count reaches past the size given, or past the reader's bounds.
	ErrPastSize = errors.New("gguf: a length past the size given")
	// ErrMalformed: a value type the format does not have, a nested array, or a tensor of more
	// than eight dimensions.
	ErrMalformed = errors.New("gguf: malformed header")
	// ErrNoTensors: Info was asked of a reading made without the tensor table.
	ErrNoTensors = errors.New("gguf: the tensor table was not read")
)

// VersionError is a GGUF version the reader does not know (it knows 2 and 3).
type VersionError struct{ Version uint32 }

func (e *VersionError) Error() string { return fmt.Sprintf("gguf: unsupported version %d", e.Version) }

// Kind is what a kept metadata value holds.
type Kind uint8

const (
	KindInt    Kind = iota + 1 // an integer of any width: Bits, read with Int or Uint
	KindBool                   // Bits is 0 or 1
	KindString                 // Str
	KindInts                   // an array of integers or bools of at most MaxIntArray elements: Arr
)

// Value is one kept metadata value as the header holds it.
type Value struct {
	Kind Kind
	// Type is the header's own type id of the value, or of an array's elements (0 uint8, 1 int8,
	// 2 uint16, 3 int16, 4 uint32, 5 int32, 7 bool, 8 string, 10 uint64, 11 int64): its width and sign.
	Type uint32
	// Bits is an integer or bool widened to 64 bits: sign-extended where Type is signed.
	Bits uint64
	Str  string
	// Arr is an array's elements, each widened as Bits is.
	Arr []uint64
}

// Signed reports whether the value's type (or its elements') is a signed integer.
func (v Value) Signed() bool {
	switch kind(v.Type) {
	case kI8, kI16, kI32, kI64:
		return true
	}
	return false
}

// Int is an integer or bool value as an int64, false for another kind or a uint64 past the range.
func (v Value) Int() (int64, bool) {
	if v.Kind != KindInt && v.Kind != KindBool {
		return 0, false
	}
	if !v.Signed() && v.Bits > math.MaxInt64 {
		return 0, false
	}
	return int64(v.Bits), true
}

// Uint is an integer or bool value as a uint64, false for another kind or a negative value.
func (v Value) Uint() (uint64, bool) {
	if v.Kind != KindInt && v.Kind != KindBool {
		return 0, false
	}
	if v.Signed() && int64(v.Bits) < 0 {
		return 0, false
	}
	return v.Bits, true
}

// number is v as Info reads it: an integer or bool as a float64.
func (v Value) number() (float64, bool) {
	if v.Kind != KindInt && v.Kind != KindBool {
		return 0, false
	}
	if v.Signed() {
		return float64(int64(v.Bits)), true
	}
	return float64(v.Bits), true
}

type reader struct {
	r    *bufio.Reader
	pos  int64
	size int64 // the size given; 0 or less: the lengths are held to the reader's bounds alone
	err  error
}

// fits refuses a read of n more bytes past the size given (ErrPastSize).
func (rd *reader) fits(n uint64) bool {
	if rd.err != nil {
		return false
	}
	if rd.size > 0 && (rd.pos > rd.size || n > uint64(rd.size-rd.pos)) {
		rd.err = fmt.Errorf("%w: %d bytes at %d of %d", ErrPastSize, n, rd.pos, rd.size)
		return false
	}
	return true
}

// past refuses an array of n elements that no size could hold.
func (rd *reader) past(n uint64) {
	rd.err = fmt.Errorf("%w: an array of %d elements", ErrPastSize, n)
}

func (rd *reader) cut(err error) {
	if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
		err = ErrCut
	}
	rd.err = err
}

func (rd *reader) bytes(n int) []byte {
	if !rd.fits(uint64(n)) {
		return nil
	}
	b := make([]byte, n)
	if _, err := io.ReadFull(rd.r, b); err != nil {
		rd.cut(err)
		return nil
	}
	rd.pos += int64(n)
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
func (rd *reader) discard(n uint64) {
	if !rd.fits(n) {
		return
	}
	for n > 0 {
		step := min(n, math.MaxInt32)
		d, err := rd.r.Discard(int(step))
		n -= uint64(d)
		rd.pos += int64(d)
		if err != nil {
			rd.cut(err)
			return
		}
	}
}

// str reads a string whole. A long one is read as it arrives, so a header that claims more than
// the stream holds allocates what arrived, not what it claimed.
func (rd *reader) str() string {
	n := rd.u64()
	if rd.err != nil {
		return ""
	}
	if n > MaxString {
		rd.err = fmt.Errorf("%w: a string of %d bytes, past the reader's %d", ErrPastSize, n, MaxString)
		return ""
	}
	if n <= 1<<16 {
		return string(rd.bytes(int(n)))
	}
	if !rd.fits(n) {
		return ""
	}
	var b bytes.Buffer
	got, err := io.CopyN(&b, rd.r, int64(n))
	rd.pos += got
	if err != nil {
		rd.cut(err)
		return ""
	}
	return b.String()
}

// int reads one integer or bool of type t, widened (sign-extended where t is signed).
func (rd *reader) int(t kind) uint64 {
	b := rd.bytes(kindSize[t])
	if b == nil {
		return 0
	}
	switch t {
	case kU8, kBool:
		return uint64(b[0])
	case kI8:
		return uint64(int64(int8(b[0])))
	case kU16:
		return uint64(binary.LittleEndian.Uint16(b))
	case kI16:
		return uint64(int64(int16(binary.LittleEndian.Uint16(b))))
	case kU32:
		return uint64(binary.LittleEndian.Uint32(b))
	case kI32:
		return uint64(int64(int32(binary.LittleEndian.Uint32(b))))
	}
	return binary.LittleEndian.Uint64(b) // kU64, kI64
}

func integer(t kind) bool { _, ok := kindSize[t]; return ok && t != kF32 && t != kF64 }

// value reads one metadata value of type t, and false where it was walked past: a float, an
// array of strings or floats, an integer array longer than MaxIntArray.
func (rd *reader) value(t kind) (Value, bool) {
	switch {
	case t == kStr:
		return Value{Kind: KindString, Type: uint32(t), Str: rd.str()}, true
	case t == kBool:
		return Value{Kind: KindBool, Type: uint32(t), Bits: rd.int(t)}, true
	case integer(t):
		return Value{Kind: KindInt, Type: uint32(t), Bits: rd.int(t)}, true
	case t == kF32 || t == kF64:
		rd.discard(uint64(kindSize[t]))
		return Value{}, false
	case t != kArr:
		rd.err = fmt.Errorf("%w: value type %d", ErrMalformed, t)
		return Value{}, false
	}
	et := kind(rd.u32())
	n := rd.u64()
	if rd.err != nil {
		return Value{}, false
	}
	switch {
	case et == kArr:
		rd.err = fmt.Errorf("%w: a nested array", ErrMalformed)
	case et == kStr:
		if n > math.MaxUint64/8 {
			rd.past(n)
		} else if rd.fits(n * 8) { // each string is at least its length
			for i := uint64(0); i < n && rd.err == nil; i++ {
				rd.discard(rd.u64())
			}
		}
	case kindSize[et] == 0:
		rd.err = fmt.Errorf("%w: array element type %d", ErrMalformed, et)
	case n > math.MaxUint64/8:
		rd.past(n)
	case !rd.fits(n * uint64(kindSize[et])): // past the size: fits has said so
	case !integer(et) || n > MaxIntArray:
		rd.discard(n * uint64(kindSize[et]))
	default:
		v := Value{Kind: KindInts, Type: uint32(et), Arr: make([]uint64, 0, n)}
		for i := uint64(0); i < n && rd.err == nil; i++ {
			v.Arr = append(v.Arr, rd.int(et))
		}
		return v, true
	}
	return Value{}, false
}

// File is one reading of a header: what Info, the chat template and every caller's own facts
// are taken from.
type File struct {
	Version uint32
	// Meta is every metadata key whose value is an integer of any width, a bool, a string, or an
	// array of integers or bools of at most MaxIntArray elements.
	Meta map[string]Value
	// Passed is how many values were walked past: floats, arrays of strings or floats, and
	// integer arrays longer than MaxIntArray.
	Passed int
	// Tensors is the tensor table, where it was asked for (nil otherwise).
	Tensors  []Tensor
	FileSize int64
	tensors  bool // whether the tensor table was read
}

// ReadHeader reads a GGUF header (v2, v3) from r, a stream at the file's start, and size, the
// file's size in bytes: the metadata always, the tensor table where tensors is true. It never
// reads past the tensor table. A header cut short is ErrCut, a wrong magic ErrMagic, an unknown
// version *VersionError, a length or count past size (or the reader's bounds) ErrPastSize, a type
// the format does not have ErrMalformed; it never allocates more than the stream gave.
func ReadHeader(r io.Reader, size int64, tensors bool) (*File, error) {
	rd := &reader{r: bufio.NewReaderSize(r, 1<<20), size: size}
	if m := rd.u32(); rd.err != nil || m != ggufMagic {
		if rd.err != nil {
			return nil, rd.err
		}
		return nil, ErrMagic
	}
	f := &File{Meta: map[string]Value{}, FileSize: size, tensors: tensors}
	f.Version = rd.u32()
	if rd.err != nil {
		return nil, rd.err
	}
	if f.Version < 2 || f.Version > 3 {
		return nil, &VersionError{Version: f.Version}
	}
	nt, nkv := rd.u64(), rd.u64()
	if rd.err != nil {
		return nil, rd.err
	}
	// A key is at least its length, its type and one byte; a tensor its length, rank, type and offset.
	if nt > maxCount || nkv > maxCount || !rd.fits(nkv*13) {
		return nil, fmt.Errorf("%w: %d tensors, %d keys", ErrPastSize, nt, nkv)
	}
	for i := uint64(0); i < nkv; i++ {
		k := rd.str()
		t := kind(rd.u32())
		if rd.err != nil {
			return nil, fmt.Errorf("gguf: metadata %d: %w", i, rd.err)
		}
		v, kept := rd.value(t)
		if rd.err != nil {
			return nil, fmt.Errorf("gguf: metadata %d (%q): %w", i, k, rd.err)
		}
		if kept {
			f.Meta[k] = v
		} else {
			f.Passed++
		}
	}
	if !tensors {
		return f, nil
	}
	if !rd.fits(nt * 24) {
		return nil, fmt.Errorf("%w: %d tensors", ErrPastSize, nt)
	}
	for i := uint64(0); i < nt; i++ {
		var t Tensor
		t.Name = rd.str()
		nd := rd.u32()
		if rd.err == nil && nd > maxRank {
			return nil, fmt.Errorf("gguf: tensor %q: %w: %d dimensions", t.Name, ErrMalformed, nd)
		}
		for d := uint32(0); d < nd && rd.err == nil; d++ {
			t.Dims = append(t.Dims, rd.u64())
		}
		t.Type = rd.u32()
		t.Offset = rd.u64()
		if rd.err != nil {
			return nil, fmt.Errorf("gguf: tensor %d: %w", i, rd.err)
		}
		if err := checkTensor(t); err != nil {
			return nil, fmt.Errorf("gguf: tensor %d: %w", i, err)
		}
		f.Tensors = append(f.Tensors, t)
	}
	f.sizeTensors()
	return f, nil
}

// Parse is ReadHeader with the tensor table: what Info needs. fileSize also sizes the last tensor
// where its type is not in the size table.
func Parse(r io.Reader, fileSize int64) (*File, error) { return ReadHeader(r, fileSize, true) }

// ChatTemplate is the model's chat template (tokenizer.chat_template) exactly as the header holds
// it, and false where the header has none.
func (f *File) ChatTemplate() (string, bool) {
	v, ok := f.Meta["tokenizer.chat_template"]
	return v.Str, ok && v.Kind == KindString
}

// ChatTemplates is the header's named templates (tokenizer.chat_template.<name>) by name; nil
// where it has none.
func (f *File) ChatTemplates() map[string]string {
	var out map[string]string
	for k, v := range f.Meta {
		name, ok := strings.CutPrefix(k, "tokenizer.chat_template.")
		if !ok || v.Kind != KindString {
			continue
		}
		if out == nil {
			out = map[string]string{}
		}
		out[name] = v.Str
	}
	return out
}

// elements is the element count of dims, and false where it passes MaxInt64 or a dimension is 0 or
// past MaxInt64.
func elements(dims []uint64) (int64, bool) {
	n := int64(1)
	for _, d := range dims {
		if d == 0 || d > math.MaxInt64 || n > math.MaxInt64/int64(d) {
			return 0, false
		}
		n *= int64(d)
	}
	return n, true
}

// tensorBytes is the size of elems elements of a type in the size table ({block elements, bytes
// per block}), rounded up to a whole block, and false where it passes MaxInt64.
func tensorBytes(elems int64, ts [2]int64) (int64, bool) {
	blocks := elems / ts[0]
	if elems%ts[0] != 0 {
		blocks++
	}
	if blocks > math.MaxInt64/ts[1] {
		return 0, false
	}
	return blocks * ts[1], true
}

// checkTensor refuses a tensor whose dimensions or offset no size could hold (ErrMalformed).
func checkTensor(t Tensor) error {
	elems, ok := elements(t.Dims)
	if !ok {
		return fmt.Errorf("%w: tensor %q: dimensions %v hold no whole count of elements", ErrMalformed, t.Name, t.Dims)
	}
	if ts, known := typeSize[t.Type]; known {
		if _, ok := tensorBytes(elems, ts); !ok {
			return fmt.Errorf("%w: tensor %q: %d elements of type %d pass the int64 range", ErrMalformed, t.Name, elems, t.Type)
		}
	}
	if t.Offset > math.MaxInt64 {
		return fmt.Errorf("%w: tensor %q: offset %d", ErrMalformed, t.Name, t.Offset)
	}
	return nil
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
			elems, _ := elements(t.Dims) // checked by checkTensor
			t.Bytes, _ = tensorBytes(elems, ts)
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

func (f *File) num(key string) (float64, bool) { return f.Meta[key].number() }

func (f *File) arr(key string) ([]float64, bool) {
	v, ok := f.Meta[key]
	if !ok || v.Kind != KindInts {
		return nil, false
	}
	out := make([]float64, len(v.Arr))
	for i, x := range v.Arr {
		out[i], _ = Value{Kind: KindInt, Type: v.Type, Bits: x}.number()
	}
	return out, true
}

// Info derives the planner's view of the model from the reading, which must hold the tensor
// table (ErrNoTensors).
func (f *File) Info() (Info, error) {
	var in Info
	if !f.tensors {
		return in, ErrNoTensors
	}
	a := f.Meta["general.architecture"].Str
	if a == "" {
		return in, errors.New("gguf: no general.architecture")
	}
	in.Arch = a
	key := func(s string) string { return a + "." + s }
	geti := func(s string) int { n, _ := f.num(key(s)); return int(n) }

	if bc, ok := f.num(key("block_count")); ok {
		if bc < 0 || bc > maxLayers { // 0 is read as not said, as before: the names tell
			return in, fmt.Errorf("%w: block_count %v, not in 0..%d", ErrMalformed, bc, maxLayers)
		}
		in.NLayer = int(bc)
	}
	var notes []string
	for _, t := range f.Tensors {
		if i, ok := blockIndex(t.Name); ok && i >= maxLayers {
			return in, fmt.Errorf("%w: tensor %q: block index past %d", ErrMalformed, t.Name, maxLayers-1)
		}
	}
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
	in.CtxTrain = geti("context_length")
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
	if in.NLayer > 0 {
		in.BlockBytes = make([]int64, in.NLayer)
	}
	var outW, outNorm, tokEmbd int64
	hasOut := false
	for _, t := range f.Tensors {
		switch t.Name {
		case "output.weight":
			outW, hasOut = t.Bytes, true
		case "output_norm.weight":
			outNorm = t.Bytes
		case "token_embd.weight":
			tokEmbd = t.Bytes
		}
		if i, ok := blockIndex(t.Name); ok && i < in.NLayer && t.Bytes >= 0 && t.Bytes <= math.MaxInt64-in.BlockBytes[i] {
			in.BlockBytes[i] += t.Bytes
		}
		if t.Bytes < 0 || t.Bytes > math.MaxInt64-in.ExpertBytes-in.DenseBytes {
			return in, fmt.Errorf("%w: the tensor sizes pass the int64 range at %q", ErrMalformed, t.Name)
		}
		if strings.Contains(t.Name, "_exps") {
			in.ExpertBytes += t.Bytes
			if i, ok := blockIndex(t.Name); ok {
				exLayers[i] = true
			}
		} else {
			in.DenseBytes += t.Bytes
			switch t.Name {
			case "per_layer_token_embd.weight":
				in.HostBytes += t.Bytes
			case "token_embd.weight":
				if len(t.Dims) == 2 {
					in.VocabSize = int(t.Dims[1])
				}
			}
		}
	}
	in.ExpertLayers = len(exLayers)
	in.OutputTied = !hasOut
	if hasOut {
		in.OutputBytes = outW + outNorm
	} else {
		in.OutputBytes = tokEmbd + outNorm
	}
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
		if n < maxLayers { // past the bound the index is only "too big": it saturates, never wraps
			n = n*10 + int(c-'0')
		}
		digits++
	}
	n = min(n, maxLayers)
	return n, digits > 0
}

// sharedKV reports whether layer i reuses an earlier layer's cache (the last n layers).
func sharedKV(nLayer, n, i int) bool { return n > 0 && n < nLayer && i >= nLayer-n }

// SmallParams is the parameter count below which a model is "small".
const SmallParams = 8_000_000_000

// ParamCount is the model's total parameter count (a mixture-of-experts model counts every
// expert): general.parameter_count when the header has it, else the sum of the tensor element
// counts. 0 when neither is known.
func (f *File) ParamCount() int64 {
	if n, ok := f.Meta["general.parameter_count"].Int(); ok && n > 0 {
		return n
	}
	var sum int64
	for _, t := range f.Tensors {
		n, ok := elements(t.Dims)
		if !ok || n > math.MaxInt64-sum {
			return 0 // past the int64 range: not known
		}
		sum += n
	}
	return sum
}

// SizeClass is "small" below SmallParams parameters, "large" at or above, "" when params is 0.
func SizeClass(params int64) string {
	switch {
	case params <= 0:
		return ""
	case params < SmallParams:
		return "small"
	}
	return "large"
}
