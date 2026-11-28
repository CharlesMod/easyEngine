package api_test

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/CharlesMod/easyEngine/catalog"
	"github.com/CharlesMod/easyEngine/internal/testkit"
	"github.com/CharlesMod/easyEngine/run"
)

// lookEnv is a ready engine with the projector loaded (serve --vision, the file present); the
// projector's detail table is low 64, high 128, max 256 and its look kwargs say enable_thinking false.
func lookEnv(t *testing.T, mut func(*run.Config)) (*testkit.Env, *run.Supervisor, string) {
	t.Helper()
	return lookEnvE(t, nil, mut, true)
}

// lookEnvE is lookEnv with a say before the engine starts (pre: the fake's env, the catalog) and
// the wait for the projector optional (wait false: the caller waits, or the engine sleeps).
func lookEnvE(t *testing.T, pre func(*testkit.Env), mut func(*run.Config), wait bool) (*testkit.Env, *run.Supervisor, string) {
	t.Helper()
	testkit.SkipWindows(t)
	e := testkit.New(t)
	e.InstallFake()
	e.Cat.Models[0].Projector = &catalog.Projector{File: visionProjFile, URL: "http://invalid.example/mm1", SHA256: strings.Repeat("ab", 32),
		Size: visionProjSize, LoadMiB: 100, Detail: catalog.Detail{Low: 64, High: 128, Max: 256},
		ChatTemplateKwargs: map[string]json.RawMessage{"enable_thinking": json.RawMessage("false")}}
	if err := os.WriteFile(filepath.Join(e.Data, "models", visionProjFile), bytes.Repeat([]byte{1}, visionProjSize), 0o644); err != nil {
		t.Fatal(err)
	}
	if pre != nil {
		pre(e)
	}
	cfg := e.Config()
	cfg.Vision = true
	if mut != nil {
		mut(&cfg)
	}
	s, srv := serve(t, e, cfg)
	if wait {
		testkit.WaitPhase(t, s, "ready", 15*time.Second)
		testkit.WaitFor(t, 20*time.Second, "vision loaded", func() bool { return s.VisionLoaded() })
	}
	return e, s, srv.URL
}

func pngB64(t *testing.T, w, h int) string {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.Set(x, y, color.RGBA{uint8(x), uint8(y), 7, 255})
		}
	}
	var b bytes.Buffer
	if err := png.Encode(&b, img); err != nil {
		t.Fatal(err)
	}
	return base64.StdEncoding.EncodeToString(b.Bytes())
}

// A local look: the engine's own chat request reaches the model server (the image re-encoded and
// scaled to the detail's edge, the label and question as text parts, the projector's kwargs),
// and the answer is the look answer with the model server's figures.
func TestLookLocal(t *testing.T) {
	e, _, base := lookEnv(t, nil)
	body := fmt.Sprintf(`{"images":[{"data":%q,"label":"the map"}],"question":"what is here?","detail":"low","max_tokens":64}`, pngB64(t, 200, 100))
	c, b, _ := do(t, "POST", base+"/engine/v1/look", body, csrf)
	if c != 200 {
		t.Fatalf("look: %d %s", c, b)
	}
	var ans struct {
		Text   string         `json:"text"`
		Tokens map[string]int `json:"tokens"`
		MS     map[string]int `json:"ms"`
	}
	if err := json.Unmarshal([]byte(b), &ans); err != nil {
		t.Fatal(err)
	}
	if ans.Text != "saw 1 images" || ans.Tokens["image"] != -1 || ans.Tokens["prompt"] < 100 || ans.Tokens["completion"] != 5 ||
		ans.MS["prefill"] != 13 || ans.MS["total"] != 113 {
		t.Fatalf("answer %s", b)
	}
	var child struct {
		Messages []struct {
			Content []struct {
				Type     string `json:"type"`
				Text     string `json:"text"`
				ImageURL struct {
					URL string `json:"url"`
				} `json:"image_url"`
			} `json:"content"`
		} `json:"messages"`
		MaxTokens int             `json:"max_tokens"`
		Kwargs    map[string]bool `json:"chat_template_kwargs"`
	}
	if err := json.Unmarshal([]byte(upstreamBytes(t, e)), &child); err != nil {
		t.Fatal(err)
	}
	parts := child.Messages[0].Content
	if len(parts) != 3 || parts[0].Text != "the map" || parts[2].Text != "what is here?" || child.MaxTokens != 64 || child.Kwargs["enable_thinking"] {
		t.Fatalf("the model server saw %s", upstreamBytes(t, e))
	}
	const pfx = "data:image/png;base64,"
	if !strings.HasPrefix(parts[1].ImageURL.URL, pfx) {
		t.Fatalf("image part %q", parts[1].ImageURL.URL[:40])
	}
	raw, _ := base64.StdEncoding.DecodeString(strings.TrimPrefix(parts[1].ImageURL.URL, pfx))
	cfg, err := png.DecodeConfig(bytes.NewReader(raw))
	if err != nil || cfg.Width != 64 || cfg.Height != 32 {
		t.Fatalf("the image the model server got: %v %dx%d (want 64x32)", err, cfg.Width, cfg.Height)
	}
}
