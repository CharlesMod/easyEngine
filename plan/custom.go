package plan

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"

	"github.com/CharlesMod/easyEngine/catalog"
	"github.com/CharlesMod/easyEngine/core/fit"
	"github.com/CharlesMod/easyEngine/core/gguf"
)

// CustomExtraMarginMiB is the room left over and above the usual margin for a file the catalog
// has no measurement of: its footprint is a header guess, and a guess that is a little low costs
// a failed load, so it plans for a card this much smaller.
const CustomExtraMarginMiB = 768

// CustomCheck is the verdict on a local GGUF file the user brought (contract A, custom/check).
type CustomCheck struct {
	OK           bool      `json:"ok"`
	Error        string    `json:"error,omitempty"`
	Path         string    `json:"path"`
	Name         string    `json:"name"`
	Size         int64     `json:"size"`
	Arch         string    `json:"arch,omitempty"`
	Params       int64     `json:"params"`
	SizeClass    string    `json:"size_class"` // "small" (< 8e9 params), "large", "" unknown
	CtxTrain     int       `json:"ctx_train"`
	Quant        string    `json:"quant,omitempty"`
	MoE          bool      `json:"moe"`
	Known        bool      `json:"known"`     // a catalog model's measured footprint matched
	Estimated    bool      `json:"estimated"` // the numbers rest on the header alone
	Fits         string    `json:"fits,omitempty"`
	Why          string    `json:"why,omitempty"`
	Limit        string    `json:"limit,omitempty"` // what stops it when Fits is "no" (the Limit* codes)
	Shape        fit.Shape `json:"shape"`
	EstVRAMMiB   int       `json:"est_vram_mib"`
	EstRAMMiB    int       `json:"est_ram_mib"`
	EstDecodeTPS float64   `json:"est_decode_tps"`
	ChatTemplate bool      `json:"chat_template"`
}

// fileTypes names llama.cpp's general.file_type values.
var fileTypes = map[int64]string{
	0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1",
	10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M",
	16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K", 19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S",
	22: "IQ3_XS", 23: "IQ3_XXS", 24: "IQ1_S", 25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M",
	28: "IQ2_S", 29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M", 32: "BF16", 36: "TQ1_0", 37: "TQ2_0",
	38: "MXFP4_MOE",
}

// QuantName is the quant of a parsed header ("" when it records none).
func QuantName(f *gguf.File) string {
	v, ok := f.Meta["general.file_type"]
	if !ok {
		return ""
	}
	n, ok := v.Int()
	if !ok {
		return ""
	}
	if s, ok := fileTypes[n]; ok {
		return s
	}
	return fmt.Sprintf("ftype %d", n)
}

// CustomKey is a cheap stand-in for a file's sha256 (path, size and mtime), so a ten GB file is
// never hashed. It keys the calibration book.
func CustomKey(path string, size, mtime int64) string {
	h := sha256.Sum256([]byte(fmt.Sprintf("%s|%d|%d", path, size, mtime)))
	return "custom-" + hex.EncodeToString(h[:12])
}

// CustomName is a file's base name without .gguf.
func CustomName(path string) string {
	return strings.TrimSuffix(filepath.Base(path), filepath.Ext(path))
}

// MatchCatalog finds a catalog model with a measured footprint whose file is the same size as
// the custom file. The catalog records no arch, params or quant, and a hash of a ten GB file is
// not cheap, so an exact byte size is the match.
func MatchCatalog(c *catalog.Catalog, size int64) (catalog.Model, bool) {
	if c == nil {
		return catalog.Model{}, false
	}
	for _, m := range c.Models {
		if m.Footprint != nil && m.Size > 0 && m.Size == size {
			return m, true
		}
	}
	return catalog.Model{}, false
}

// CheckCustom reads the header of path and judges it in room. It never copies or changes the
// file. c may be nil.
func CheckCustom(path string, c *catalog.Catalog, room fit.Room, w fit.Wants) CustomCheck {
	out := CustomCheck{Path: path, Name: CustomName(path)}
	if path == "" {
		out.Error = "no path given"
		return out
	}
	fh, err := os.Open(path)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			out.Error = "file not found"
		} else {
			out.Error = "file unreadable: " + err.Error()
		}
		return out
	}
	defer fh.Close()
	st, err := fh.Stat()
	if err != nil || st.IsDir() {
		out.Error = "file not found"
		return out
	}
	out.Size = st.Size()
	f, err := gguf.Parse(fh, st.Size())
	if err != nil {
		var ve *gguf.VersionError
		if errors.Is(err, gguf.ErrMagic) || errors.Is(err, gguf.ErrCut) && st.Size() < 4 || errors.As(err, &ve) {
			out.Error = "not a GGUF file"
		} else {
			out.Error = "header unreadable: " + err.Error()
		}
		return out
	}
	info, err := f.Info()
	if err != nil {
		out.Error = "header unreadable: " + err.Error()
		return out
	}
	out.Arch, out.CtxTrain, out.MoE = info.Arch, info.CtxTrain, info.NExpert > 0
	out.Quant = QuantName(f)
	_, out.ChatTemplate = f.ChatTemplate()
	out.Params = f.ParamCount()
	out.SizeClass = gguf.SizeClass(out.Params)
	out.OK = true

	w = w.Norm()
	fp, err := Seed(info, w.KVType)
	if err != nil {
		out.Fits, out.Why, out.Limit = "no", err.Error(), LimitModel
		if errors.As(err, new(*fit.KVTypeError)) {
			out.Limit = LimitKVType
		}
		return out
	}
	m, known := MatchCatalog(c, st.Size())
	if known {
		if mf, err := FromMeasured(m.Footprint, w.KVType); err == nil {
			fp.TPS = mf.TPS
		}
		room.SpeedScale = SpeedScaleFor(room.GPUName, m.Footprint)
	} else {
		out.Estimated = true
		room = WithCustomMargin(room)
	}
	out.Known = known
	sh, err := fit.Choose(fp, room, w)
	switch {
	case err != nil:
		out.Fits = "no"
		out.Limit, out.Why = whyNot(fp, room, w, err)
	default:
		out.Shape, out.EstVRAMMiB, out.EstRAMMiB, out.EstDecodeTPS = sh, sh.VRAMMiB, sh.RAMMiB, sh.EstTPS
		out.Fits, out.Why = verdictFor(fp, sh, w)
	}
	out.Estimated = out.Estimated || room.NoGPU
	return out
}

// WithCustomMargin adds CustomExtraMarginMiB to a room with a card.
func WithCustomMargin(r fit.Room) fit.Room {
	if !r.NoGPU {
		r.Margin += CustomExtraMarginMiB
	}
	return r
}

// MatchCatalogOrNil is MatchCatalog as a pointer.
func MatchCatalogOrNil(c *catalog.Catalog, size int64) *catalog.Model {
	if m, ok := MatchCatalog(c, size); ok {
		return &m
	}
	return nil
}
