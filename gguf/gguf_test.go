package gguf

import (
	"bytes"
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"testing"
)

// ---- a tiny GGUF writer for tests ----

type w struct{ bytes.Buffer }

func (b *w) u32(v uint32) { binary.Write(b, binary.LittleEndian, v) }
func (b *w) u64(v uint64) { binary.Write(b, binary.LittleEndian, v) }
func (b *w) str(s string) { b.u64(uint64(len(s))); b.WriteString(s) }

type kv struct {
	k string
	v any
}

func writeKV(b *w, e kv) {
	b.str(e.k)
	switch v := e.v.(type) {
	case string:
		b.u32(uint32(kStr))
		b.str(v)
	case uint32:
		b.u32(uint32(kU32))
		b.u32(v)
	case float32:
		b.u32(uint32(kF32))
		b.u32(math.Float32bits(v))
	case []uint32:
		b.u32(uint32(kArr))
		b.u32(uint32(kU32))
		b.u64(uint64(len(v)))
		for _, x := range v {
			b.u32(x)
		}
	case []bool:
		b.u32(uint32(kArr))
		b.u32(uint32(kBool))
		b.u64(uint64(len(v)))
		for _, x := range v {
			if x {
				b.WriteByte(1)
			} else {
				b.WriteByte(0)
			}
		}
	case []string:
		b.u32(uint32(kArr))
		b.u32(uint32(kStr))
		b.u64(uint64(len(v)))
		for _, x := range v {
			b.str(x)
		}
	default:
		panic("unsupported")
	}
}

type tn struct {
	name string
	dims []uint64
	typ  uint32
}

func build(version uint32, kvs []kv, ts []tn) []byte {
	var b w
	b.u32(ggufMagic)
	b.u32(version)
	b.u64(uint64(len(ts)))
	b.u64(uint64(len(kvs)))
	for _, e := range kvs {
		writeKV(&b, e)
	}
	var off uint64
	for _, t := range ts {
		b.str(t.name)
		b.u32(uint32(len(t.dims)))
		for _, d := range t.dims {
			b.u64(d)
		}
		b.u32(t.typ)
		b.u64(off)
		ty := typeSize[t.typ]
		el := uint64(1)
		for _, d := range t.dims {
			el *= d
		}
		off += el / uint64(ty[0]) * uint64(ty[1])
		off = (off + 31) &^ 31
	}
	return b.Bytes()
}

func writeFile(t *testing.T, data []byte) string {
	p := filepath.Join(t.TempDir(), "m.gguf")
	if err := os.WriteFile(p, data, 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

func gemmaLike() []byte {
	pat := make([]bool, 6)
	for i := range pat {
		pat[i] = i%3 != 2 // layers 2 and 5 are full attention
	}
	kvs := []kv{
		{"general.architecture", "gemma4"},
		{"tokenizer.ggml.tokens", []string{"a", "bb", "ccc"}},
		{"gemma4.block_count", uint32(6)},
		{"gemma4.attention.head_count", uint32(16)},
		{"gemma4.attention.head_count_kv", []uint32{8, 8, 2, 8, 8, 2}},
		{"gemma4.attention.key_length", uint32(512)},
		{"gemma4.attention.value_length", uint32(512)},
		{"gemma4.attention.key_length_swa", uint32(256)},
		{"gemma4.attention.value_length_swa", uint32(256)},
		{"gemma4.attention.sliding_window", uint32(1024)},
		{"gemma4.attention.sliding_window_pattern", pat},
		{"gemma4.expert_count", uint32(8)},
		{"gemma4.expert_used_count", uint32(2)},
		{"gemma4.attention.layer_norm_rms_epsilon", float32(1e-6)},
	}
	var ts []tn
	ts = append(ts, tn{"token_embd.weight", []uint64{256, 100}, 8}) // 100*256/32*34
	for i := 0; i < 6; i++ {
		p := "blk." + string(rune('0'+i)) + "."
		ts = append(ts, tn{p + "attn_q.weight", []uint64{256, 64}, 12}, // 256-elt Q4_K rows
			tn{p + "ffn_gate_up_exps.weight", []uint64{256, 32, 8}, 18},
			tn{p + "ffn_down_exps.weight", []uint64{32, 256, 8}, 20},
			tn{p + "ffn_down_exps.scale", []uint64{8}, 0})
	}
	return build(3, kvs, ts)
}

func TestGemmaLike(t *testing.T) {
	data := gemmaLike()
	p := writeFile(t, data)
	in, err := Read(p)
	if err != nil {
		t.Fatal(err)
	}
	if in.Arch != "gemma4" || in.NLayer != 6 || in.SlidingWindow != 1024 || in.SWALayers != 4 || in.NExpert != 8 || in.NExpertUsed != 2 {
		t.Errorf("%+v", in)
	}
	// scalar KV fields describe full-attention layers
	if in.NKVHead != 2 || in.HeadDimK != 512 || in.HeadDimV != 512 || in.HeadDimKSWA != 256 {
		t.Errorf("kv %+v", in)
	}
	if len(in.Layers) != 6 || !in.Layers[0].SWA || in.Layers[2].SWA || in.Layers[0].KVHeads != 8 || in.Layers[0].HeadDimK != 256 || in.Layers[2].HeadDimK != 512 {
		t.Errorf("layers %+v", in.Layers)
	}
	wantExp := int64(6 * (256*32*8/256*98 + 32*256*8/32*18 + 8*4))
	if in.ExpertLayers != 6 {
		t.Errorf("expert layers %d", in.ExpertLayers)
	}
	// ffn_down_exps.scale is F32 and also named _exps: it counts as expert bytes
	if in.ExpertBytes != wantExp {
		t.Errorf("expert bytes %d want %d", in.ExpertBytes, wantExp)
	}
	wantDense := int64(100*256/32*34 + 6*(256*64/256*144))
	if in.DenseBytes != wantDense {
		t.Errorf("dense bytes %d want %d", in.DenseBytes, wantDense)
	}
	if in.FileSize != int64(len(data)) {
		t.Errorf("file size %d", in.FileSize)
	}
}

func TestDenseLlamaScalarKV(t *testing.T) {
	kvs := []kv{
		{"general.architecture", "llama"},
		{"llama.block_count", uint32(32)},
		{"llama.embedding_length", uint32(4096)},
		{"llama.attention.head_count", uint32(32)},
		{"llama.attention.head_count_kv", uint32(8)},
	}
	p := writeFile(t, build(3, kvs, []tn{{"blk.0.attn_q.weight", []uint64{256, 4}, 12}}))
	in, err := Read(p)
	if err != nil {
		t.Fatal(err)
	}
	if in.NKVHead != 8 || in.HeadDimK != 128 || in.HeadDimV != 128 || in.SWALayers != 0 || in.NExpert != 0 || in.ExpertBytes != 0 || in.ExpertLayers != 0 {
		t.Errorf("%+v", in)
	}
	if len(in.Layers) != 32 || in.Layers[31].KVHeads != 8 || in.Layers[31].SWA || !in.Layers[31].HasKV {
		t.Errorf("layers")
	}
}

func TestGemma3PeriodAndSharedKV(t *testing.T) {
	kvs := []kv{
		{"general.architecture", "gemma3n"},
		{"gemma3n.block_count", uint32(12)},
		{"gemma3n.embedding_length", uint32(2048)},
		{"gemma3n.attention.head_count", uint32(8)},
		{"gemma3n.attention.head_count_kv", uint32(2)},
		{"gemma3n.attention.key_length", uint32(256)},
		{"gemma3n.attention.sliding_window", uint32(512)},
		{"gemma3n.attention.sliding_window_pattern", uint32(5)},
		{"gemma3n.attention.shared_kv_layers", uint32(4)},
	}
	in, err := Read(writeFile(t, build(3, kvs, nil)))
	if err != nil {
		t.Fatal(err)
	}
	// layers 4 and 9 are full attention (every 5th); the last 4 layers share an earlier cache
	if in.SWALayers != 10 || in.Layers[4].SWA || in.Layers[9].SWA || in.Layers[8].HasKV != false || in.Layers[7].HasKV != true {
		t.Errorf("%+v", in.Layers)
	}
	if in.HeadDimV != 256 {
		t.Errorf("value length should default to key length, got %d", in.HeadDimV)
	}
}

func TestHybridFullAttentionInterval(t *testing.T) {
	kvs := []kv{
		{"general.architecture", "qwen3next"},
		{"qwen3next.block_count", uint32(8)},
		{"qwen3next.attention.head_count", uint32(16)},
		{"qwen3next.attention.head_count_kv", uint32(2)},
		{"qwen3next.attention.key_length", uint32(256)},
		{"qwen3next.full_attention_interval", uint32(4)},
	}
	in, err := Read(writeFile(t, build(3, kvs, nil)))
	if err != nil {
		t.Fatal(err)
	}
	n := 0
	for _, l := range in.Layers {
		if l.HasKV {
			n++
		}
	}
	if n != 2 || !in.Layers[3].HasKV || in.Layers[0].HasKV {
		t.Errorf("%+v", in.Layers)
	}
}

func TestBadFiles(t *testing.T) {
	if _, err := Read(filepath.Join(t.TempDir(), "nope")); err == nil {
		t.Error("missing file")
	}
	if _, err := Read(writeFile(t, []byte("not a gguf file at all"))); err == nil {
		t.Error("bad magic")
	}
	good := gemmaLike()
	if _, err := Read(writeFile(t, good[:len(good)/2])); err == nil {
		t.Error("truncated")
	}
	v1 := append([]byte(nil), good...)
	binary.LittleEndian.PutUint32(v1[4:], 1)
	if _, err := Read(writeFile(t, v1)); err == nil {
		t.Error("v1")
	}
	if _, err := Read(writeFile(t, build(3, []kv{{"general.name", "x"}}, nil))); err == nil {
		t.Error("no architecture")
	}
}

// The real reference model: header only, read-only. Skipped where the file does not exist.
func TestReferenceModel(t *testing.T) {
	home, _ := os.UserHomeDir()
	p := filepath.Join(home, ".hive/models/gemma4-rp-maxxed/gemma4-rp-maxxed-Q3_K_XL.gguf")
	if _, err := os.Stat(p); err != nil {
		t.Skip("reference model not on this machine")
	}
	in, err := Read(p)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("%+v", in)
	if in.Arch != "gemma4" || in.NLayer != 30 || in.SlidingWindow != 1024 || in.SWALayers != 25 || in.NExpert != 96 || in.NExpertUsed != 8 {
		t.Errorf("%+v", in)
	}
	if in.NKVHead != 2 || in.HeadDimK != 512 || in.HeadDimV != 512 || in.HeadDimKSWA != 256 || in.Layers[0].KVHeads != 8 {
		t.Errorf("kv %+v", in)
	}
	if in.ExpertLayers != 30 || in.ExpertBytes <= 0 || in.DenseBytes <= 0 {
		t.Errorf("bytes %+v", in)
	}
	// tensor bytes plus the header must account for (almost) the whole file
	if d := in.FileSize - in.ExpertBytes - in.DenseBytes; d < 0 || d > 32<<20 {
		t.Errorf("tensor bytes %d vs file %d (gap %d)", in.ExpertBytes+in.DenseBytes, in.FileSize, d)
	}
}

// lingLike mirrors the header keys of Ling 3.0-tiny (bailingmoe3): MLA attention every fourth
// layer (head_count_kv 1), Kimi delta attention (head_count_kv 0, fixed state) elsewhere.
func lingLike() []byte {
	kvh := make([]uint32, 24)
	for i := 3; i < 24; i += 4 {
		kvh[i] = 1
	}
	kvs := []kv{
		{"general.architecture", "bailingmoe3"},
		{"bailingmoe3.block_count", uint32(24)},
		{"bailingmoe3.embedding_length", uint32(1536)},
		{"bailingmoe3.attention.head_count", uint32(16)},
		{"bailingmoe3.attention.head_count_kv", kvh},
		{"bailingmoe3.attention.key_length", uint32(576)},
		{"bailingmoe3.attention.value_length", uint32(128)},
		{"bailingmoe3.attention.kv_lora_rank", uint32(512)},
		{"bailingmoe3.rope.dimension_count", uint32(64)},
		{"bailingmoe3.kda.head_dim", uint32(128)},
		{"bailingmoe3.ssm.conv_kernel", uint32(4)},
		{"bailingmoe3.expert_count", uint32(128)},
		{"bailingmoe3.expert_used_count", uint32(8)},
	}
	var ts []tn
	for i := 1; i < 24; i++ {
		ts = append(ts, tn{name: "blk." + itoa(i) + ".ffn_up_exps.weight", dims: []uint64{1536, 512, 128}, typ: 12})
	}
	ts = append(ts, tn{name: "token_embd.weight", dims: []uint64{1536, 1000}, typ: 12})
	return build(3, kvs, ts)
}

func itoa(i int) string { return strconv.Itoa(i) }

func TestMLAAndLinearAttention(t *testing.T) {
	f, err := Parse(bytes.NewReader(lingLike()), 5<<30)
	if err != nil {
		t.Fatal(err)
	}
	in, err := f.Info()
	if err != nil {
		t.Fatal(err)
	}
	full := 0
	for i, l := range in.Layers {
		if l.HasKV {
			full++
			if i%4 != 3 || l.KVHeads != 1 || l.HeadDimK != 576 || l.HeadDimV != 0 {
				t.Errorf("layer %d: %+v", i, l)
			}
		} else if !l.Recurrent {
			t.Errorf("layer %d should be recurrent: %+v", i, l)
		}
	}
	if full != 6 || in.KVLoraRank != 512 || in.NKVHead != 1 || in.HeadDimK != 576 || in.HeadDimV != 0 {
		t.Errorf("%+v", in)
	}
	// 18 delta-attention layers: (3*3*16*128 + 128*128*16) f32 each
	if want := int64(18 * (3*3*16*128 + 128*128*16) * 4); in.StateBytesPerSlot != want {
		t.Errorf("state %d want %d", in.StateBytesPerSlot, want)
	}
	if in.Estimated || in.ExpertLayers != 23 {
		t.Errorf("%+v", in)
	}
}

func TestUnknownArchitectureIsEstimated(t *testing.T) {
	data := build(3, []kv{{"general.architecture", "mystery"}}, []tn{
		{name: "blk.0.attn_q.weight", dims: []uint64{64, 64}, typ: 0},
		{name: "blk.1.attn_q.weight", dims: []uint64{64, 64}, typ: 0},
	})
	f, err := Parse(bytes.NewReader(data), 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	in, err := f.Info()
	if err != nil {
		t.Fatal(err)
	}
	if !in.Estimated || in.Note == "" || in.NLayer != 2 || len(in.Layers) != 2 {
		t.Fatalf("%+v", in)
	}
	for _, l := range in.Layers {
		if !l.HasKV || l.KVHeads <= 0 || l.HeadDimK <= 0 {
			t.Errorf("estimate must never be zero: %+v", l)
		}
	}
	// no layer count anywhere: still an estimate
	f2, _ := Parse(bytes.NewReader(build(3, []kv{{"general.architecture", "mystery"}}, nil)), 1<<20)
	in2, err := f2.Info()
	if err != nil || in2.NLayer != 32 || !in2.Estimated {
		t.Errorf("%+v %v", in2, err)
	}
}

// The real Ling 3.0-tiny header (first 8 MiB of the Q4_K_M file, fetched by Range request).
// Skipped where it is not on disk.
func TestLingRealHeader(t *testing.T) {
	home, _ := os.UserHomeDir()
	p := filepath.Join(home, ".local/share", "rim"+"world-hermes-engine", "tmp/ling-head.bin")
	fh, err := os.Open(p)
	if err != nil {
		t.Skip("Ling header not on this machine")
	}
	defer fh.Close()
	f, err := Parse(fh, 4820000000)
	if err != nil {
		t.Fatal(err)
	}
	in, err := f.Info()
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("%+v", in)
	if in.Arch != "bailingmoe3" || in.NLayer != 24 || in.NExpert != 128 || in.NExpertUsed != 8 || in.KVLoraRank != 512 {
		t.Errorf("%+v", in)
	}
	full := 0
	for i, l := range in.Layers {
		if l.HasKV {
			full++
			if i%4 != 3 || l.HeadDimK != 576 || l.KVHeads != 1 || l.HeadDimV != 0 {
				t.Errorf("layer %d %+v", i, l)
			}
		}
	}
	if full != 6 || in.ExpertLayers != 23 || in.Estimated {
		t.Errorf("full=%d %+v", full, in)
	}
}
