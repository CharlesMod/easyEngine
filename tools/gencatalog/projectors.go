package main

import (
	"encoding/json"
	"fmt"
	"net/url"
	"regexp"
	"sort"
	"strings"

	"github.com/CharlesMod/easyEngine/catalog"
)

// projSpec says which mmproj file is a model's vision projector and what the engine measured for
// it. The projector is pinned to a full commit: its url, size and hash describe one revision.
type projSpec struct {
	repo, rev, file string // rev: a full 40-hex commit sha, never a branch
	local           string // file name in the models directory
	loadMiB         int
	imageMiB        int
	maxImageTokens  int
	detail          catalog.Detail
	kwargs          map[string]json.RawMessage // chat_template_kwargs for look calls (Bonfire: enable_thinking false)
}

// projectorSpecs is keyed by catalog model id.
//
// Proposal, not an entry yet (Bonfire, the hive's Qwen 27B): repo ggml-org/Qwen3.8-27B-GGUF,
// file mmproj-Qwen3.8-27B-Q8_0.gguf, rev c1869413 (short: resolve it to the full sha when it is
// added), loadMiB 852, imageMiB 106 (958 peak - 852), maxImageTokens 0, detail low 768, high 1536,
// max 2048, kwargs {"enable_thinking": false} (each look call must turn thinking off).
var projectorSpecs = map[string]projSpec{
	"gemma4-rp-maxxed-q3kxl": {
		repo: "unsloth/gemma-4-26B-A4B-it-GGUF", rev: "c099eb48e663fd284577b04978a94ffccb261841",
		file: "mmproj-F16.gguf", local: "gemma-4-26B-A4B-it-mmproj-F16.gguf",
		loadMiB: 1296, imageMiB: 130, maxImageTokens: 512,
		detail: catalog.Detail{Low: 512, High: 768, Max: 1024},
	},
}

var fullRev = regexp.MustCompile(`^[0-9a-f]{40}$`)

// applyProjectors attaches every spec's projector to its model in cat. A spec for a model the
// catalog lacks, a short rev, a file missing from the tree or one without LFS metadata fails the run.
func applyProjectorsTo(c *client, models []catalog.Model) error {
	cat := catalog.Catalog{Models: models} // the same backing array: the projectors land in models
	return applyProjectorSpecs(c, &cat, projectorSpecs)
}

func applyProjectorSpecs(c *client, cat *catalog.Catalog, specs map[string]projSpec) error {
	ids := make([]string, 0, len(specs))
	for id := range specs {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	for _, id := range ids {
		idx := -1
		for i, m := range cat.Models {
			if m.ID == id {
				idx = i
			}
		}
		if idx < 0 {
			return fmt.Errorf("projector spec for model %q, which is not in the catalog", id)
		}
		p, err := c.hfProjector(specs[id])
		if err != nil {
			return fmt.Errorf("projector of %s: %w", id, err)
		}
		c.log("projector %s %d bytes", p.File, p.Size)
		cat.Models[idx].Projector = &p
	}
	return nil
}

func (c *client) hfProjector(sp projSpec) (catalog.Projector, error) {
	if !fullRev.MatchString(sp.rev) {
		return catalog.Projector{}, fmt.Errorf("%s: rev %q is not a full 40-hex commit sha", sp.repo, sp.rev)
	}
	var tree []hfEntry
	if err := c.getJSON(c.hfBase()+"/api/models/"+sp.repo+"/tree/"+sp.rev+"?limit=1000", &tree); err != nil {
		return catalog.Projector{}, err
	}
	for _, e := range tree {
		if e.Path != sp.file {
			continue
		}
		if e.LFS == nil || len(e.LFS.OID) != 64 {
			return catalog.Projector{}, fmt.Errorf("%s/%s: no LFS sha256", sp.repo, sp.file)
		}
		size := e.LFS.Size
		if size == 0 {
			size = e.Size
		}
		return catalog.Projector{
			File:   sp.local,
			URL:    c.hfBase() + "/" + sp.repo + "/resolve/" + sp.rev + "/" + url.PathEscape(sp.file),
			SHA256: strings.ToLower(e.LFS.OID), Size: size,
			LoadMiB: sp.loadMiB, ImageMiB: sp.imageMiB, MaxImageTokens: sp.maxImageTokens, Detail: sp.detail,
			ChatTemplateKwargs: sp.kwargs,
		}, nil
	}
	return catalog.Projector{}, fmt.Errorf("%s: file %s not in the repo at %s", sp.repo, sp.file, sp.rev)
}
