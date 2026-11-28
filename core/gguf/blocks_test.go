package gguf

import (
	"reflect"
	"testing"
)

// denseBlocks is a small dense model: three blocks of different weight, the token table, the
// per-layer token table the E-series keeps in RAM, the output norm and, where untied, an output head.
func denseBlocks(untied bool) []byte {
	kvs := []kv{
		{"general.architecture", "llama"},
		{"llama.block_count", uint32(3)},
		{"llama.embedding_length", uint32(256)},
		{"llama.attention.head_count", uint32(4)},
		{"llama.attention.head_count_kv", uint32(2)},
	}
	ts := []tn{
		{"token_embd.weight", []uint64{256, 100}, 8},          // Q8_0: 100*256/32*34 = 27200
		{"per_layer_token_embd.weight", []uint64{256, 10}, 0}, // F32: 10240
		{"blk.0.attn_q.weight", []uint64{256, 64}, 0},         // 65536
		{"blk.0.ffn_up.weight", []uint64{256, 8}, 0},          // 8192
		{"blk.1.attn_q.weight", []uint64{256, 32}, 0},         // 32768
		{"blk.2.attn_q.weight", []uint64{256, 16}, 0},         // 16384
		{"output_norm.weight", []uint64{256}, 0},              // 1024
	}
	if untied {
		ts = append(ts, tn{"output.weight", []uint64{256, 100}, 8}) // 27200
	}
	return build(3, kvs, ts)
}

// Info gives each block's bytes (what -ngl moves with that block) and the output head that goes to
// the card with any block: output.weight and its norm, or where the file ties the head to the token
// table (no output.weight: llama.cpp reads token_embd again as the head, gemma4.cpp 55-59), the
// token table and the norm. Tensors outside a block are in none, and every byte is still in
// DenseBytes as before.
func TestBlockAndOutputBytes(t *testing.T) {
	for _, untied := range []bool{false, true} {
		in, err := read(writeFile(t, denseBlocks(untied)))
		if err != nil {
			t.Fatal(err)
		}
		if want := []int64{65536 + 8192, 32768, 16384}; !reflect.DeepEqual(in.BlockBytes, want) {
			t.Errorf("untied %v: blocks %v, want %v", untied, in.BlockBytes, want)
		}
		wantOut, wantDense := int64(27200+1024), int64(27200+10240+65536+8192+32768+16384+1024)
		if untied {
			wantDense += 27200
		}
		if in.OutputTied == untied || in.OutputBytes != wantOut {
			t.Errorf("untied %v: output %d tied %v, want %d", untied, in.OutputBytes, in.OutputTied, wantOut)
		}
		if in.DenseBytes != wantDense || in.HostBytes != 10240 {
			t.Errorf("untied %v: dense %d host %d, want %d and 10240", untied, in.DenseBytes, in.HostBytes, wantDense)
		}
	}
	// A header naming no block tensor: one entry per layer it reads (the reader's guess of the
	// count stands), each of no bytes, never a missing slice a caller would take for no header.
	p := writeFile(t, build(3, []kv{{"general.architecture", "llama"}}, []tn{{"token_embd.weight", []uint64{256, 4}, 0}}))
	in, err := read(p)
	if err != nil || len(in.BlockBytes) != in.NLayer {
		t.Fatalf("no block tensors: %d entries for %d layers (%v)", len(in.BlockBytes), in.NLayer, err)
	}
	for i, b := range in.BlockBytes {
		if b != 0 {
			t.Errorf("no block tensors: block %d holds %d bytes", i, b)
		}
	}
}
