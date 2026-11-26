package gguf

import (
	"bytes"
	"testing"
)

func TestParamCountAndSizeClass(t *testing.T) {
	ts := []tn{{"token_embd.weight", []uint64{256, 100}, 0}, {"blk.0.attn_q.weight", []uint64{256, 256}, 0}}
	parse := func(kvs []kv) *File {
		f, err := Parse(bytes.NewReader(build(3, kvs, ts)), 1<<30)
		if err != nil {
			t.Fatal(err)
		}
		return f
	}
	arch := kv{"general.architecture", "llama"}
	if n := parse([]kv{arch}).ParamCount(); n != 256*100+256*256 {
		t.Fatalf("tensor sum = %d", n)
	}
	if n := parse([]kv{arch, {"general.parameter_count", uint32(7)}}).ParamCount(); n != 7 {
		t.Fatalf("header count = %d", n)
	}
	for _, c := range []struct {
		n    int64
		want string
	}{{0, ""}, {-1, ""}, {7_999_999_999, "small"}, {8_000_000_000, "large"}, {35_000_000_000, "large"}} {
		if got := SizeClass(c.n); got != c.want {
			t.Errorf("SizeClass(%d) = %q, want %q", c.n, got, c.want)
		}
	}
}
