// STUB: replaced by eng/fetch
package catalog

import (
	"encoding/json"
	"hermesengine/probe"
	"os"
	"path/filepath"
)

type Asset struct {
	ID, OS, Arch, Accel, URL, SHA256 string
	Size                             int64
	Extra                            []string
}
type Model struct {
	ID, Name, File, URL, SHA256 string
	Size                        int64
	MoE                         bool
	Quality                     int
	Chat, Notes                 string
}
type LlamaCpp struct {
	Tag    string
	Assets []Asset
}
type Catalog struct {
	Version  int
	LlamaCpp LlamaCpp
	UV       []Asset
	Models   []Model
}

func Load(dataDir string) (*Catalog, error) {
	c := &Catalog{Version: 1}
	if b, err := os.ReadFile(filepath.Join(dataDir, "catalog.json")); err == nil {
		var n Catalog
		if json.Unmarshal(b, &n) == nil && n.Version > c.Version {
			return &n, nil
		}
	}
	return c, nil
}

func (c *Catalog) Model(id string) (Model, bool) {
	for _, m := range c.Models {
		if m.ID == id {
			return m, true
		}
	}
	return Model{}, false
}

func (c *Catalog) Asset(id string) (Asset, bool) {
	for _, a := range c.LlamaCpp.Assets {
		if a.ID == id {
			return a, true
		}
	}
	return Asset{}, false
}

func (c *Catalog) Backends(hw probe.Hardware) []Asset {
	nv := false
	for _, g := range hw.GPUs {
		if g.Vendor == "nvidia" {
			nv = true
		}
	}
	var out []Asset
	for _, acc := range []string{"cuda", "metal", "vulkan", "cpu"} {
		if acc == "cuda" && !nv {
			continue
		}
		for _, a := range c.LlamaCpp.Assets {
			if a.Accel == acc && a.OS == hw.OS && a.Arch == hw.Arch {
				out = append(out, a)
			}
		}
	}
	return out
}
