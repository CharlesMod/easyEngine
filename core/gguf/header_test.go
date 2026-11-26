package gguf

import (
	"bytes"
	"encoding/binary"
	"errors"
	"math"
	"reflect"
	"runtime"
	"strings"
	"testing"
)

func neg(v int64) uint64 { return uint64(v) }

// oddWidths is a header holding a value of every kind: integers of each width and sign, a bool,
// a string, short arrays of integers and bools, and what is walked past (two floats, an array of
// strings, an integer array past MaxIntArray, an array of floats), with the chat template and a
// named one. widths is the same with longN elements in the long array, for a test that reads
// every prefix.
func oddWidths(template string) []byte { return widths(template, MaxIntArray+1) }

func widths(template string, longN int) []byte {
	long := make([]uint64, longN)
	kvs := []kv{
		{"general.architecture", "geo"},
		{"geo.u8", num{kU8, 7}},
		{"geo.i8", num{kI8, neg(-3)}},
		{"geo.u16", num{kU16, 300}},
		{"geo.i16", num{kI16, neg(-300)}},
		{"geo.i32", num{kI32, neg(-5)}},
		{"geo.u64", num{kU64, 1 << 40}},
		{"geo.i64", num{kI64, neg(-1 << 40)}},
		{"geo.flag", num{kBool, 1}},
		{"geo.f32", float32(1e-6)},
		{"geo.f64", num{kF64, math.Float64bits(0.5)}},
		{"tokenizer.ggml.tokens", []string{"a", "bb"}},
		{"geo.attention.head_count_kv", []uint32{4, 4, 2, 2}},
		{"geo.attention.sliding_window_pattern", []bool{true, false}},
		{"tokenizer.ggml.token_type", arr{kI32, long}},
		{"geo.scores", arr{kF32, []uint64{0, 0, 0}}},
		{"tokenizer.chat_template.tool_use", "{{ tools }}"},
	}
	if template != "" {
		kvs = append(kvs, kv{"tokenizer.chat_template", template})
	}
	return build(3, kvs, []tn{{"token_embd.weight", []uint64{256, 100}, 8}, {"blk.0.ffn_up_exps.weight", []uint64{256, 4, 8}, 12}})
}

// The general reading: every kept value as the header holds it, what was walked past counted,
// and the tensor table where asked. Info is a reading of the same File, never a second walk.
func TestHeaderReading(t *testing.T) {
	data := oddWidths("{{ messages }}")
	f, err := ReadHeader(bytes.NewReader(data), int64(len(data)), true)
	if err != nil {
		t.Fatal(err)
	}
	ints := map[string]int64{"geo.u8": 7, "geo.i8": -3, "geo.u16": 300, "geo.i16": -300, "geo.i32": -5, "geo.u64": 1 << 40, "geo.i64": -1 << 40}
	for k, want := range ints {
		v := f.Meta[k]
		if got, ok := v.Int(); v.Kind != KindInt || !ok || got != want {
			t.Errorf("%s: %+v, want %d", k, v, want)
		}
		if u, ok := v.Uint(); ok != (want >= 0) || (ok && int64(u) != want) {
			t.Errorf("%s as unsigned: %d %v", k, u, ok)
		}
	}
	if v := f.Meta["geo.flag"]; v.Kind != KindBool || v.Bits != 1 {
		t.Errorf("bool: %+v", v)
	}
	if v := f.Meta["geo.attention.head_count_kv"]; v.Kind != KindInts || !reflect.DeepEqual(v.Arr, []uint64{4, 4, 2, 2}) {
		t.Errorf("integer array: %+v", v)
	}
	if v := f.Meta["geo.attention.sliding_window_pattern"]; v.Kind != KindInts || !reflect.DeepEqual(v.Arr, []uint64{1, 0}) {
		t.Errorf("bool array: %+v", v)
	}
	for _, k := range []string{"geo.f32", "geo.f64", "tokenizer.ggml.tokens", "tokenizer.ggml.token_type", "geo.scores"} {
		if _, ok := f.Meta[k]; ok {
			t.Errorf("%s is kept; it is walked past", k)
		}
	}
	if f.Passed != 5 {
		t.Errorf("walked past %d values, want 5", f.Passed)
	}
	if len(f.Tensors) != 2 || f.Tensors[1].Name != "blk.0.ffn_up_exps.weight" || !reflect.DeepEqual(f.Tensors[1].Dims, []uint64{256, 4, 8}) ||
		f.Tensors[1].Type != 12 || f.Tensors[1].Bytes != 256*4*8/256*144 {
		t.Errorf("tensors: %+v", f.Tensors)
	}

	meta, err := ReadHeader(bytes.NewReader(data), int64(len(data)), false)
	if err != nil || meta.Tensors != nil || !reflect.DeepEqual(meta.Meta, f.Meta) || meta.Passed != f.Passed {
		t.Fatalf("the metadata alone: %v", err)
	}
	if _, err := meta.Info(); !errors.Is(err, ErrNoTensors) {
		t.Errorf("Info of a reading without the tensor table: %v", err)
	}
	if _, err := f.Info(); err != nil {
		t.Errorf("Info: %v", err)
	}
}

// The chat template as the header holds it, whole at any length, and the named ones; a header
// with none says so.
func TestChatTemplate(t *testing.T) {
	long := strings.Repeat("{% for m in messages %}{{ m.content }}{% endfor %}", 4000) // about 200 KiB
	data := oddWidths(long)
	f, err := ReadHeader(bytes.NewReader(data), int64(len(data)), false)
	if err != nil {
		t.Fatal(err)
	}
	if got, ok := f.ChatTemplate(); !ok || got != long {
		t.Errorf("the template: %d bytes (%v), want %d", len(got), ok, len(long))
	}
	if got := f.ChatTemplates(); !reflect.DeepEqual(got, map[string]string{"tool_use": "{{ tools }}"}) {
		t.Errorf("named templates: %v", got)
	}
	none := oddWidths("")
	if f, err = ReadHeader(bytes.NewReader(none), int64(len(none)), false); err != nil {
		t.Fatal(err)
	}
	if got, ok := f.ChatTemplate(); ok || got != "" {
		t.Errorf("a header with no template answered %q", got)
	}
}

// A header cut short, with the wrong magic, a version the reader does not know, or a length past
// the size given, is a typed error: never a panic, never an allocation of what the header claims.
func TestHeaderRefusals(t *testing.T) {
	good := widths("{{ messages }}", 3)
	size := int64(len(good))
	for n := 0; n < len(good); n++ {
		if _, err := ReadHeader(bytes.NewReader(good[:n]), size, true); !errors.Is(err, ErrCut) {
			t.Fatalf("cut at %d of %d: %v", n, len(good), err)
		}
	}
	bad := append([]byte("GGUX"), good[4:]...)
	if _, err := ReadHeader(bytes.NewReader(bad), size, true); !errors.Is(err, ErrMagic) {
		t.Errorf("wrong magic: %v", err)
	}
	for _, v := range []uint32{1, 4} {
		other := append([]byte(nil), good...)
		binary.LittleEndian.PutUint32(other[4:], v)
		var ve *VersionError
		if _, err := ReadHeader(bytes.NewReader(other), size, true); !errors.As(err, &ve) || ve.Version != v {
			t.Errorf("version %d: %v", v, err)
		}
	}
	// A key's length past the size given; the template's past it; an array's past it.
	past := func(name string, kvs []kv, size int64) {
		data := build(3, kvs, nil)
		if size == 0 {
			size = int64(len(data))
		}
		if _, err := ReadHeader(bytes.NewReader(data), size, true); !errors.Is(err, ErrPastSize) {
			t.Errorf("%s: %v", name, err)
		}
	}
	tmpl := strings.Repeat("x", 4096)
	past("a template past the size", []kv{{"tokenizer.chat_template", tmpl}}, 4000)
	past("an integer array past the size", []kv{{"a", arr{kU32, make([]uint64, 2000)}}}, 4000)
	past("a string array past the size", []kv{{"a", []string{tmpl}}}, 4000)
	counts := build(3, nil, nil)
	binary.LittleEndian.PutUint64(counts[16:], 1000) // a thousand keys in 24 bytes
	if _, err := ReadHeader(bytes.NewReader(counts), int64(len(counts)), true); !errors.Is(err, ErrPastSize) {
		t.Errorf("a count past the size: %v", err)
	}

	// A header that claims a string of 200 MiB in a stream that ends, with a size that allows it:
	// cut short, and what was allocated is what arrived, not what was claimed.
	var b w
	b.u32(ggufMagic)
	b.u32(3)
	b.u64(0)
	b.u64(1)
	b.str("tokenizer.chat_template")
	b.u32(uint32(kStr))
	b.u64(200 << 20)
	b.WriteString("{{")
	var before, after runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&before)
	_, err := ReadHeader(bytes.NewReader(b.Bytes()), 1<<40, true)
	runtime.ReadMemStats(&after)
	if !errors.Is(err, ErrCut) {
		t.Errorf("a claimed 200 MiB string, cut: %v", err)
	}
	if grew := after.TotalAlloc - before.TotalAlloc; grew > 8<<20 {
		t.Errorf("reading a cut header allocated %d bytes", grew)
	}
	// Past the reader's own bound for one string: refused, never cut to fit.
	b.Reset()
	b.u32(ggufMagic)
	b.u32(3)
	b.u64(0)
	b.u64(1)
	b.str("tokenizer.chat_template")
	b.u32(uint32(kStr))
	b.u64(MaxString + 1)
	if _, err := ReadHeader(bytes.NewReader(b.Bytes()), 1<<40, true); !errors.Is(err, ErrPastSize) {
		t.Errorf("a string past the reader's bound: %v", err)
	}
}

// Parity with the other program's reader, from the same synthetic headers its tests build (a
// dense model with geometry, a float and a general key; a mixture of experts with stacked expert
// tensors) and the template and odd widths above. What that reader gives its callers, and where
// each is in this reading:
//
//   - the version: File.Version
//   - every key whose value is an integer of any width, a bool, a string, or a short array of
//     integers or bools: File.Meta (Int, Uint, Bits, Str, Arr); floats and other arrays walked past
//   - the tensor table, each tensor's name, dimensions, type and offset: File.Tensors
//   - a tensor's parameters (the product of its dimensions): from Tensor.Dims
//   - the architecture and the size label: Meta "general.architecture", "general.size_label"
//   - the geometry keys as stated (an integer, or an array of one per layer): Meta, by key
//   - the expert count and the experts used: Meta "<arch>.expert_count", "<arch>.expert_used_count"
//   - dense or mixture of experts, from the expert count or stacked expert tensors ("_exps"),
//     the parameters in total and active, and the bits per weight from the file's size: worked
//     out below from Meta, Tensors and the size given (the core reads; it does not classify)
//   - the chat template: File.ChatTemplate
//
// Not given in the same form: a negative signed integer, which that reader gives as its unsigned
// bits at the type's width (an int32 -5 as 4294967291), is given here as its value (Int: -5); a
// caller wanting those bits masks Bits to the width of Value.Type.
func TestParityWithTheOtherReader(t *testing.T) {
	dense := build(3, []kv{
		{"general.architecture", "geo"},
		{"geo.block_count", uint32(4)},
		{"geo.attention.head_count_kv", []uint32{4, 4, 2, 2}},
		{"geo.rope.freq_base", float32(10000)},
		{"general.context_length", uint32(9)},
		{"general.size_label", "7B"},
	}, []tn{{"blk.0.attn_q.weight", []uint64{100, 500}, 0}})
	f, err := ReadHeader(bytes.NewReader(dense), int64(len(dense)), true)
	if err != nil {
		t.Fatal(err)
	}
	fa := factsOf(f, 50_000)
	if f.Version != 3 || fa.arch != "geo" || fa.label != "7B" || fa.moe || fa.total != 50_000 || fa.active != 50_000 || fa.bpw != 8 {
		t.Errorf("dense: %+v", fa)
	}
	if n, _ := f.Meta["geo.block_count"].Int(); n != 4 || !reflect.DeepEqual(f.Meta["geo.attention.head_count_kv"].Arr, []uint64{4, 4, 2, 2}) {
		t.Errorf("dense geometry: %+v", f.Meta)
	}
	if n, _ := f.Meta["general.context_length"].Int(); n != 9 || f.Passed != 1 {
		t.Errorf("a general key kept and the float walked past: %+v, passed %d", f.Meta["general.context_length"], f.Passed)
	}
	if f.Tensors[0].Offset != 0 || f.Tensors[0].Type != 0 {
		t.Errorf("tensor: %+v", f.Tensors[0])
	}

	moe := build(3, []kv{
		{"general.architecture", "mix"},
		{"mix.expert_count", uint32(8)},
		{"mix.expert_used_count", uint32(2)},
	}, []tn{{"token_embd.weight", []uint64{100, 200}, 0}, {"blk.0.ffn_gate_exps.weight", []uint64{8, 25, 100}, 0}})
	if f, err = ReadHeader(bytes.NewReader(moe), int64(len(moe)), true); err != nil {
		t.Fatal(err)
	}
	if fa := factsOf(f, 0); !fa.moe || fa.experts != 8 || fa.used != 2 || fa.total != 40_000 || fa.active != 20_000+5_000 || fa.bpw != 0 {
		t.Errorf("moe: %+v", fa)
	}

	odd := oddWidths("{{ messages }}")
	if f, err = ReadHeader(bytes.NewReader(odd), int64(len(odd)), true); err != nil {
		t.Fatal(err)
	}
	if tmpl, ok := f.ChatTemplate(); !ok || tmpl != "{{ messages }}" {
		t.Errorf("template %q", tmpl)
	}
	if v := f.Meta["geo.i32"]; v.Bits&0xFFFFFFFF != 4294967291 || v.Type != uint32(kI32) {
		t.Errorf("an int32's bits at its width: %+v", v)
	}
	if v := f.Meta["geo.flag"]; v.Bits != 1 {
		t.Errorf("a bool as 1: %+v", v)
	}
}

// facts is what the other program works out of a header, worked out here from the core's reading.
type facts struct {
	arch, label           string
	moe                   bool
	experts, used         int64
	total, active, expert int64
	bpw                   float64
}

func factsOf(f *File, size int64) facts {
	fa := facts{arch: f.Meta["general.architecture"].Str, label: f.Meta["general.size_label"].Str}
	fa.experts, _ = f.Meta[fa.arch+".expert_count"].Int()
	fa.used, _ = f.Meta[fa.arch+".expert_used_count"].Int()
	stacked := false
	for _, t := range f.Tensors {
		n := int64(1)
		for _, d := range t.Dims {
			n *= int64(d)
		}
		fa.total += n
		if strings.Contains(t.Name, "_exps") {
			fa.expert += n
			stacked = true
		}
	}
	fa.moe = fa.experts > 0 || stacked
	fa.active = fa.total
	if fa.moe && fa.experts > 0 && fa.used > 0 {
		fa.active = fa.total - fa.expert + int64(float64(fa.expert)*float64(fa.used)/float64(fa.experts))
	}
	if fa.total > 0 && size > 0 {
		fa.bpw = float64(size) * 8 / float64(fa.total)
	}
	return fa
}
