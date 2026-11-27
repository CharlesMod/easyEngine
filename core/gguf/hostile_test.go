package gguf

import (
	"bytes"
	"errors"
	"math"
	"testing"
)

// hostile parses data as a caller does (the size is the data's length) and derives Info, and
// fails the test on a panic. It returns the first error.
func hostile(t *testing.T, data []byte) (err error) {
	t.Helper()
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("panic on a hostile header: %v", r)
		}
	}()
	f, err := Parse(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		return err
	}
	in, err := f.Info()
	if err != nil {
		return err
	}
	if in.DenseBytes < 0 || in.ExpertBytes < 0 || in.HostBytes < 0 || in.NLayer < 0 {
		t.Fatalf("negative size: %+v", in)
	}
	return nil
}

func withArch(extra []kv) []kv {
	return append([]kv{{"general.architecture", "llama"}}, extra...)
}

func TestHostileBlockCount(t *testing.T) {
	cases := map[string]num{
		"huge u64":     {kU64, 1 << 40},
		"huge i64":     {kI64, 1 << 62},
		"max u64":      {kU64, math.MaxUint64},
		"negative i32": {kI32, neg(-5)},
		"negative i64": {kI64, neg(math.MinInt64)},
		"just past":    {kU32, maxLayers + 1},
	}
	for name, n := range cases {
		t.Run(name, func(t *testing.T) {
			data := build(3, withArch([]kv{{"llama.block_count", n}}), nil)
			if err := hostile(t, data); !errors.Is(err, ErrMalformed) {
				t.Fatalf("want ErrMalformed, got %v", err)
			}
		})
	}
	// the bound itself is fine, and 0 is read as not said (the tensor names tell), as before
	for _, n := range []uint64{maxLayers, 0} {
		data := build(3, withArch([]kv{{"llama.block_count", num{kU32, n}}}), nil)
		if err := hostile(t, data); err != nil {
			t.Fatalf("block_count %d refused: %v", n, err)
		}
	}
}

func TestHostileBlockIndex(t *testing.T) {
	up := func(name string) []tn { return []tn{{name, []uint64{4, 4}, 0}} }
	cases := map[string][]byte{
		"overflowing index, no block_count":   build(3, withArch(nil), up("blk.99999999999999999999.ffn_up.weight")),
		"overflowing index, with block_count": build(3, withArch([]kv{{"llama.block_count", uint32(4)}}), up("blk.99999999999999999999.ffn_up.weight")),
		"wrapping index":                      build(3, withArch(nil), up("blk.18446744073709551616.ffn_up.weight")),
		"index past the bound, no count":      build(3, withArch(nil), up("blk.1000000.ffn_up.weight")),
		"expert index past the bound":         build(3, withArch([]kv{{"llama.block_count", uint32(4)}}), up("blk.65536.ffn_up_exps.weight")),
	}
	for name, data := range cases {
		t.Run(name, func(t *testing.T) {
			if err := hostile(t, data); !errors.Is(err, ErrMalformed) {
				t.Fatalf("want ErrMalformed, got %v", err)
			}
		})
	}
	// an index under the bound is kept, even past block_count (nextn/MTP blocks): nothing is sized
	// from it then
	for _, n := range []string{"blk.3.ffn_up.weight", "blk.4.ffn_up.weight", "blk.65535.ffn_up_exps.weight"} {
		data := build(3, withArch([]kv{{"llama.block_count", uint32(4)}}), up(n))
		if err := hostile(t, data); err != nil {
			t.Fatalf("%s of 4 refused: %v", n, err)
		}
	}
}

func TestHostileTensorSizes(t *testing.T) {
	one := func(dims []uint64, typ uint32) []byte {
		return build(3, withArch(nil), []tn{{"output.weight", dims, typ}})
	}
	cases := map[string][]byte{
		"elements past int64":        one([]uint64{1 << 32, 1 << 32}, 24),
		"elements wrap to small":     one([]uint64{1 << 63, 2}, 24),
		"elements wrap to zero":      one([]uint64{1 << 32, 1 << 32, 1 << 32}, 24),
		"bytes past int64":           one([]uint64{1 << 62}, 1),
		"dimension past int64":       one([]uint64{math.MaxUint64}, 24),
		"negative dimension":         one([]uint64{neg(-1), 4}, 0),
		"zero dimension":             one([]uint64{4, 0}, 0),
		"sum of tensors past int64":  build(3, withArch(nil), []tn{{"a.weight", []uint64{1 << 62}, 24}, {"b.weight", []uint64{1 << 62}, 24}}),
		"expert sum past int64":      build(3, withArch(nil), []tn{{"a_exps.weight", []uint64{1 << 62}, 24}, {"b_exps.weight", []uint64{1 << 62}, 24}}),
		"offset past int64, unknown": rawUnknownType(1 << 63),
		"offset max uint64, unknown": rawUnknownType(math.MaxUint64),
	}
	for name, data := range cases {
		t.Run(name, func(t *testing.T) {
			if err := hostile(t, data); !errors.Is(err, ErrMalformed) {
				t.Fatalf("want ErrMalformed, got %v", err)
			}
		})
	}
}

// rawUnknownType is a header with one tensor of a type the size table lacks, at offset off.
func rawUnknownType(off uint64) []byte {
	var b w
	b.u32(ggufMagic)
	b.u32(3)
	b.u64(1)
	b.u64(1)
	writeKV(&b, kv{"general.architecture", "llama"})
	b.str("output.weight")
	b.u32(1)
	b.u64(8)
	b.u32(99)
	b.u64(off)
	return b.Bytes()
}

func TestHostileParamCount(t *testing.T) {
	f := &File{Tensors: []Tensor{{Dims: []uint64{1 << 40, 1 << 40}}}}
	if n := f.ParamCount(); n != 0 {
		t.Fatalf("overflowing parameter count = %d, want 0", n)
	}
	f = &File{Tensors: []Tensor{{Dims: []uint64{1 << 62}}, {Dims: []uint64{1 << 62}}}}
	if n := f.ParamCount(); n != 0 {
		t.Fatalf("overflowing parameter sum = %d, want 0", n)
	}
}
