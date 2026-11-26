package testkit

import (
	"bytes"
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"
)

// WriteGGUF writes a tiny, valid dense GGUF (arch "llama", 2 layers, Q4_K_M file type, with or
// without a chat template) at dir/name and returns its path. The tensors carry no data: only the
// header is ever read.
func WriteGGUF(t testing.TB, dir, name string, chatTemplate bool) string {
	t.Helper()
	var b bytes.Buffer
	u32 := func(v uint32) { binary.Write(&b, binary.LittleEndian, v) }
	u64 := func(v uint64) { binary.Write(&b, binary.LittleEndian, v) }
	str := func(s string) { u64(uint64(len(s))); b.WriteString(s) }
	type kv struct {
		k string
		v any
	}
	kvs := []kv{
		{"general.architecture", "llama"},
		{"general.file_type", uint32(15)},
		{"llama.block_count", uint32(2)},
		{"llama.context_length", uint32(8192)},
		{"llama.embedding_length", uint32(256)},
		{"llama.attention.head_count", uint32(4)},
		{"llama.attention.head_count_kv", uint32(2)},
		{"llama.attention.key_length", uint32(64)},
		{"llama.attention.value_length", uint32(64)},
		{"tokenizer.ggml.tokens", []string{"a", "bb", "ccc"}},
	}
	if chatTemplate {
		kvs = append(kvs, kv{"tokenizer.chat_template", "{{ messages }}"})
	}
	type tn struct {
		name string
		dims []uint64
	}
	ts := []tn{{"token_embd.weight", []uint64{256, 100}}}
	for _, p := range []string{"blk.0.", "blk.1."} {
		ts = append(ts, tn{p + "attn_q.weight", []uint64{256, 256}}, tn{p + "ffn_down.weight", []uint64{512, 256}})
	}
	u32(0x46554747)
	u32(3)
	u64(uint64(len(ts)))
	u64(uint64(len(kvs)))
	for _, e := range kvs {
		str(e.k)
		switch v := e.v.(type) {
		case string:
			u32(8)
			str(v)
		case uint32:
			u32(4)
			u32(v)
		case []string:
			u32(9)
			u32(8)
			u64(uint64(len(v)))
			for _, x := range v {
				str(x)
			}
		}
	}
	var off uint64
	for _, x := range ts {
		str(x.name)
		u32(uint32(len(x.dims)))
		el := uint64(1)
		for _, d := range x.dims {
			u64(d)
			el *= d
		}
		u32(0) // F32
		u64(off)
		off += el * 4
		off = (off + 31) &^ 31
	}
	for b.Len()%32 != 0 {
		b.WriteByte(0)
	}
	b.Write(make([]byte, off))
	p := filepath.Join(dir, name)
	if err := os.WriteFile(p, b.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}
