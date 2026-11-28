package catalog

import "fmt"

// Projector is a model's vision projector: an mmproj GGUF that llama-server loads beside the
// model (--mmproj) so the model can read images. It is downloaded like a model file, into the
// same models directory, and only when vision is wanted (docs/ENGINE.md, "Vision").
type Projector struct {
	File   string `json:"file"`   // local file name in the models directory (plain name)
	URL    string `json:"url"`    // pinned to a commit; empty: import-only
	SHA256 string `json:"sha256"` // of the whole file
	Size   int64  `json:"size"`   // bytes
	// LoadMiB is what loading the projector adds on the card; ImageMiB is what the first image
	// adds on top (the encoder's scratch). The fit is given that much less room when the
	// projector is to be loaded.
	LoadMiB  int `json:"load_mib"`
	ImageMiB int `json:"image_mib"`
	// MaxImageTokens is the most tokens one image becomes under the engine's batch settings
	// (Gemma 4 at -ub 512: 512). 0: the engine's settings do not cap it (Qwen: about W/32*H/32).
	MaxImageTokens int `json:"max_image_tokens"`
	// Detail is the long edge, in pixels, an image is scaled down to for each detail level of
	// the look call (never up).
	Detail Detail `json:"detail"`
}

// Detail holds the long-edge pixels for each detail level.
type Detail struct {
	Low  int `json:"low"`
	High int `json:"high"`
	Max  int `json:"max"`
}

// MaxDetailPx bounds a detail table's long edge: nothing bigger is ever sent to a projector.
const MaxDetailPx = 8192

// RoomMiB is what the fit gives up on the card when the projector is to be loaded.
func (p *Projector) RoomMiB() int {
	if p == nil {
		return 0
	}
	return p.LoadMiB + p.ImageMiB
}

func checkProjector(m Model) error {
	p := m.Projector
	if p == nil {
		return nil
	}
	if err := checkName("projector file", p.File); err != nil {
		return fmt.Errorf("model %q: %w", m.ID, err)
	}
	if p.File == m.File {
		return fmt.Errorf("model %q: projector file is the model file", m.ID)
	}
	if err := checkDownload("projector of model", m.ID, p.URL, p.SHA256, p.Size); err != nil {
		return err
	}
	if p.LoadMiB < 0 || p.ImageMiB < 0 || p.MaxImageTokens < 0 {
		return fmt.Errorf("model %q: projector load_mib, image_mib and max_image_tokens must not be negative", m.ID)
	}
	d := p.Detail
	if d.Low <= 0 || d.Low > d.High || d.High > d.Max || d.Max > MaxDetailPx {
		return fmt.Errorf("model %q: projector detail must be 0 < low <= high <= max <= %d px", m.ID, MaxDetailPx)
	}
	return nil
}

// checkFiles refuses a catalog whose files collide: no two models share a file, no projector is
// any model's file, and two models share a projector file only when it is the same download.
func checkFiles(models []Model) error {
	files := map[string]string{}
	projs := map[string]Model{}
	for _, m := range models {
		if id, ok := files[m.File]; ok {
			return fmt.Errorf("model file %q is already %q's", m.File, id)
		}
		files[m.File] = m.ID
	}
	for _, m := range models {
		p := m.Projector
		if p == nil {
			continue
		}
		if id, ok := files[p.File]; ok {
			return fmt.Errorf("projector file %q of model %q is model %q's file", p.File, m.ID, id)
		}
		if o, ok := projs[p.File]; ok && (o.Projector.SHA256 != p.SHA256 || o.Projector.Size != p.Size || o.Projector.URL != p.URL) {
			return fmt.Errorf("projector file %q of model %q is a different file for model %q", p.File, m.ID, o.ID)
		}
		projs[p.File] = m
	}
	return nil
}

// ProjectorFor returns the projector of the model whose file or projector file is name, and
// whether name is that projector (for storage listings).
func (c *Catalog) ProjectorFor(name string) (Model, bool) {
	for _, m := range c.Models {
		if m.Projector != nil && m.Projector.File == name {
			return m, true
		}
	}
	return Model{}, false
}
