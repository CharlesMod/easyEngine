package api_test

import (
	"bufio"
	"bytes"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"hash/crc32"
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"io"
	"math/rand"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/CharlesMod/easyEngine/catalog"
	"github.com/CharlesMod/easyEngine/internal/testkit"
	"github.com/CharlesMod/easyEngine/lookimg"
	"github.com/CharlesMod/easyEngine/run"
)

// The look call's cases (docs/ENGINE.md section 7h): every refusal of the body and of an image,
// vision_off, the bounds, the caller's slots, the paired scope, the request the model server is
// given, the answer's figures and the stream. Each waits on facts (a phase, a file, a line read),
// never on a clock.

func postLook(t *testing.T, base, body string, hdr map[string]string) (int, string) {
	t.Helper()
	h := map[string]string{"Content-Type": "application/json", "X-Easy-Engine": "1"}
	for k, v := range hdr {
		h[k] = v
	}
	c, b, _ := do(t, "POST", base+"/engine/v1/look", body, h)
	return c, b
}

func jpegB64(t *testing.T, w, h int) string {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.Set(x, y, color.RGBA{uint8(x), uint8(y), 99, 255})
		}
	}
	var b bytes.Buffer
	if err := jpeg.Encode(&b, img, nil); err != nil {
		t.Fatal(err)
	}
	return base64.StdEncoding.EncodeToString(b.Bytes())
}

func oneLook(img, extra string) string {
	return fmt.Sprintf(`{"images":[{"data":%q}],"question":"q"%s}`, img, extra)
}

func pngChunk(typ string, data []byte) []byte {
	var b bytes.Buffer
	binary.Write(&b, binary.BigEndian, uint32(len(data)))
	b.WriteString(typ)
	b.Write(data)
	binary.Write(&b, binary.BigEndian, crc32.ChecksumIEEE(append([]byte(typ), data...)))
	return b.Bytes()
}

// hugePNGB64 is a header-only PNG that claims w x h.
func hugePNGB64(w, h uint32) string {
	ihdr := make([]byte, 13)
	binary.BigEndian.PutUint32(ihdr[0:], w)
	binary.BigEndian.PutUint32(ihdr[4:], h)
	ihdr[8], ihdr[9] = 8, 2
	out := []byte("\x89PNG\r\n\x1a\n")
	out = append(out, pngChunk("IHDR", ihdr)...)
	out = append(out, pngChunk("IDAT", []byte{0x78, 0x9c, 0, 0})...)
	out = append(out, pngChunk("IEND", nil)...)
	return base64.StdEncoding.EncodeToString(out)
}

// pngWithText is a w x h PNG with a tEXt chunk, written at another compression than the engine's:
// its bytes cannot come back from the engine as they went in.
func pngWithText(t *testing.T, w, h int) string {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.Set(x, y, color.RGBA{uint8(x * 5), uint8(y * 9), 7, 255})
		}
	}
	var b bytes.Buffer
	if err := (&png.Encoder{CompressionLevel: png.BestCompression}).Encode(&b, img); err != nil {
		t.Fatal(err)
	}
	raw := b.Bytes()
	at := 8 + 8 + 13 + 4 // after the signature and the IHDR chunk
	out := append(append(append([]byte{}, raw[:at]...), pngChunk("tEXt", []byte("Comment\x00secret"))...), raw[at:]...)
	return base64.StdEncoding.EncodeToString(out)
}

type childPart struct {
	Type     string `json:"type"`
	Text     string `json:"text"`
	ImageURL struct {
		URL string `json:"url"`
	} `json:"image_url"`
}

// childOf is the last request body the model server read, as the look parts of it.
func childOf(t *testing.T, e *testkit.Env) (map[string]json.RawMessage, []childPart) {
	t.Helper()
	var top map[string]json.RawMessage
	if err := json.Unmarshal([]byte(upstreamBytes(t, e)), &top); err != nil {
		t.Fatalf("the model server's last body: %v", err)
	}
	var msgs []struct {
		Role    string      `json:"role"`
		Content []childPart `json:"content"`
	}
	if err := json.Unmarshal(top["messages"], &msgs); err != nil || len(msgs) != 1 || msgs[0].Role != "user" {
		t.Fatalf("messages: %s %v", top["messages"], err)
	}
	return top, msgs[0].Content
}

// decodedPart is the image of an image part: its mime, its size and its bytes.
func decodedPart(t *testing.T, p childPart) (mime string, w, h int, raw []byte) {
	t.Helper()
	u := p.ImageURL.URL
	i := strings.Index(u, ";base64,")
	if !strings.HasPrefix(u, "data:") || i < 0 {
		t.Fatalf("image url %.40q", u)
	}
	mime = u[len("data:"):i]
	raw, err := base64.StdEncoding.DecodeString(u[i+len(";base64,"):])
	if err != nil {
		t.Fatal(err)
	}
	cfg, _, err := image.DecodeConfig(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	return mime, cfg.Width, cfg.Height, raw
}

// Every refusal of a look body is a 400 with its code, and none of them reaches the model server.
func TestLookBodyRefusals(t *testing.T) {
	e, _, base := lookEnv(t, nil)
	img := pngB64(t, 8, 8)
	valid := oneLook(img, "")
	imgs := func(n int) string {
		one := fmt.Sprintf(`{"data":%q}`, img)
		return `{"images":[` + strings.TrimSuffix(strings.Repeat(one+",", n), ",") + `],"question":"q"}`
	}
	cases := []struct{ name, body, code, msg string }{
		{"not JSON", `nope`, "bad_request", ""},
		{"empty body", ``, "bad_request", ""},
		{"trailing data", valid + ` {}`, "bad_request", "after its object"},
		{"trailing garbage", valid + `x`, "bad_request", "after its object"},
		{"array body", `[]`, "bad_request", ""},
		{"nested deeper than 8", `{"question":"q","images":[[[[[[[[1]]]]]]]]}`, "bad_request", "nests"},
		{"unknown key", oneLook(img, `,"foo":1`), "bad_request", "unknown field"},
		{"caller chat_template_kwargs", oneLook(img, `,"chat_template_kwargs":{"enable_thinking":true}`), "bad_request", "unknown field"},
		{"caller messages", oneLook(img, `,"messages":[]`), "bad_request", "unknown field"},
		{"duplicate question", fmt.Sprintf(`{"images":[{"data":%q}],"question":"a","question":"b"}`, img), "bad_request", "twice"},
		{"duplicate images", fmt.Sprintf(`{"images":[{"data":%q}],"images":[{"data":%q}],"question":"a"}`, img, img), "bad_request", "twice"},
		{"duplicate key in an image", fmt.Sprintf(`{"images":[{"data":%q,"data":%q}],"question":"q"}`, img, img), "bad_request", "twice"},
		{"unknown key in an image", fmt.Sprintf(`{"images":[{"data":%q,"url":"http://x"}],"question":"q"}`, img), "bad_request", "unknown field"},
		{"image not an object", `{"images":["x"],"question":"q"}`, "bad_request", "must be an object"},
		{"image without data", `{"images":[{"label":"x"}],"question":"q"}`, "bad_request", "required"},
		{"image data not a string", `{"images":[{"data":5}],"question":"q"}`, "bad_request", "base64 string"},
		{"images missing", `{"question":"q"}`, "bad_request", "images is required"},
		{"images not an array", `{"images":"x","question":"q"}`, "bad_request", "array"},
		{"images null", `{"images":null,"question":"q"}`, "bad_request", "array"},
		{"images empty", `{"images":[],"question":"q"}`, "bad_request", "at least one"},
		{"five images", imgs(5), "too_many_images", ""},
		{"question missing", fmt.Sprintf(`{"images":[{"data":%q}]}`, img), "bad_request", "question is required"},
		{"question empty", fmt.Sprintf(`{"images":[{"data":%q}],"question":""}`, img), "bad_request", "question is required"},
		{"question blank", fmt.Sprintf(`{"images":[{"data":%q}],"question":"  \n"}`, img), "bad_request", "question is required"},
		{"question a number", fmt.Sprintf(`{"images":[{"data":%q}],"question":5}`, img), "bad_request", "must be a string"},
		{"question too long", fmt.Sprintf(`{"images":[{"data":%q}],"question":%q}`, img, strings.Repeat("q", 16<<10+1)), "bad_request", "longer than"},
		{"label a number", fmt.Sprintf(`{"images":[{"data":%q,"label":5}],"question":"q"}`, img), "bad_request", "label must be a string"},
		{"label too long", fmt.Sprintf(`{"images":[{"data":%q,"label":%q}],"question":"q"}`, img, strings.Repeat("l", 257)), "bad_request", "longer than"},
		{"detail unknown", oneLook(img, `,"detail":"huge"`), "bad_request", "detail"},
		{"detail a number", oneLook(img, `,"detail":1`), "bad_request", "detail"},
		{"detail empty", oneLook(img, `,"detail":""`), "bad_request", "detail"},
		{"max_tokens 0", oneLook(img, `,"max_tokens":0`), "bad_request", "max_tokens"},
		{"max_tokens 4097", oneLook(img, `,"max_tokens":4097`), "bad_request", "max_tokens"},
		{"max_tokens 1.5", oneLook(img, `,"max_tokens":1.5`), "bad_request", "max_tokens"},
		{"max_tokens 1e2", oneLook(img, `,"max_tokens":1e2`), "bad_request", "max_tokens"},
		{"max_tokens a string", oneLook(img, `,"max_tokens":"5"`), "bad_request", "max_tokens"},
		{"max_tokens -1", oneLook(img, `,"max_tokens":-1`), "bad_request", "max_tokens"},
		{"max_tokens 01", oneLook(img, `,"max_tokens":01`), "bad_request", ""},
		{"max_tokens huge", oneLook(img, `,"max_tokens":99999999999`), "bad_request", "max_tokens"},
		{"max_tokens null", oneLook(img, `,"max_tokens":null`), "bad_request", "max_tokens"},
		{"id_slot -1", oneLook(img, `,"id_slot":-1`), "bad_request", "id_slot"},
		{"id_slot 1.0", oneLook(img, `,"id_slot":1.0`), "bad_request", "id_slot"},
		{"id_slot 1024", oneLook(img, `,"id_slot":1024`), "bad_request", "id_slot"},
		{"id_slot a string", oneLook(img, `,"id_slot":"0"`), "bad_request", "id_slot"},
		{"stream a string", oneLook(img, `,"stream":"true"`), "bad_request", "stream"},
		{"stream a number", oneLook(img, `,"stream":1`), "bad_request", "stream"},
	}
	before := upstreamBytes(t, e)
	for _, c := range cases {
		code, body := postLook(t, base, c.body, nil)
		if code != 400 || codeOf(body) != c.code || !strings.Contains(body, c.msg) {
			t.Errorf("%s: %d %s (want 400 %s %q)", c.name, code, body, c.code, c.msg)
		}
	}
	if after := upstreamBytes(t, e); after != before {
		t.Fatalf("a refused look reached the model server: %s", after)
	}
	if n := len(e.Starts()); n != 1 {
		t.Fatalf("starts: %d", n)
	}
}

// max_tokens 4096 is the most a look may ask, and it is what the model server is given.
func TestLookMaxTokensAccepted(t *testing.T) {
	e, _, base := lookEnv(t, nil)
	if c, b := postLook(t, base, oneLook(pngB64(t, 8, 8), `,"max_tokens":4096`), nil); c != 200 {
		t.Fatalf("max_tokens 4096: %d %s", c, b)
	}
	top, _ := childOf(t, e)
	if string(top["max_tokens"]) != "4096" {
		t.Fatalf("max_tokens %s", top["max_tokens"])
	}
	if c, b := postLook(t, base, oneLook(pngB64(t, 8, 8), ""), nil); c != 200 {
		t.Fatalf("default: %d %s", c, b)
	}
	if top, _ = childOf(t, e); string(top["max_tokens"]) != "512" {
		t.Fatalf("default max_tokens %s", top["max_tokens"])
	}
}

// Each image refusal comes with the image's index and the code, and nothing reaches the model server.
func TestLookImageRefusals(t *testing.T) {
	e, _, base := lookEnv(t, nil)
	good := pngB64(t, 8, 8)
	var gif bytes.Buffer
	gif.WriteString("GIF89a\x01\x00\x01\x00\x00\x00\x00;")
	tooLarge := strings.Repeat("A", lookimg.Default.MaxBase64+1)
	two := func(second string) string {
		return fmt.Sprintf(`{"images":[{"data":%q},{"data":%q}],"question":"q"}`, good, second)
	}
	cases := []struct {
		name, second, code string
	}{
		{"invalid base64", "@@@@not base64@@@@", "image_bad"},
		{"gif", base64.StdEncoding.EncodeToString(gif.Bytes()), "image_format"},
		{"other data prefix", "data:image/gif;base64,AAAA", "image_format"},
		{"huge header", hugePNGB64(100000, 100000), "image_too_many_pixels"},
		{"too many pixels in one side", hugePNGB64(9000, 1), "image_too_many_pixels"},
		{"base64 too long", tooLarge, "image_too_large"},
		{"truncated png", base64.StdEncoding.EncodeToString([]byte("\x89PNG\r\n\x1a\n")), "image_bad"},
	}
	before := upstreamBytes(t, e)
	for _, c := range cases {
		// the second image is the bad one: the answer says index 1
		code, body := postLook(t, base, two(c.second), nil)
		var m struct {
			Code  string `json:"code"`
			Image *int   `json:"image"`
		}
		json.Unmarshal([]byte(body), &m)
		if code != 400 || m.Code != c.code || m.Image == nil || *m.Image != 1 {
			t.Errorf("%s: %d %.300s (want 400 %s image 1)", c.name, code, body, c.code)
		}
	}
	code, body := postLook(t, base, oneLook("@@@@", ""), nil)
	var m struct {
		Image *int `json:"image"`
	}
	json.Unmarshal([]byte(body), &m)
	if code != 400 || m.Image == nil || *m.Image != 0 {
		t.Errorf("first image: %d %s", code, body)
	}
	if after := upstreamBytes(t, e); after != before {
		t.Fatalf("a refused look reached the model server: %s", after)
	}
}

// vision_off: no projector loaded on the running model (vision not wanted), or the model has none.
func TestLookVisionOffNotWanted(t *testing.T) {
	e, s, base := lookEnvE(t, nil, func(c *run.Config) { c.Vision = false }, false)
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	if s.VisionLoaded() {
		t.Fatal("vision loaded though not wanted")
	}
	c, b := postLook(t, base, oneLook(pngB64(t, 8, 8), ""), nil)
	if c != 409 || codeOf(b) != "vision_off" {
		t.Fatalf("look: %d %s", c, b)
	}
	c, b = postLook(t, base, oneLook(pngB64(t, 8, 8), `,"stream":true`), nil)
	if c != 409 || codeOf(b) != "vision_off" {
		t.Fatalf("streamed look: %d %s", c, b)
	}
	if upstreamBytes(t, e) != "" {
		t.Fatalf("the model server was asked: %s", upstreamBytes(t, e))
	}
}

func TestLookVisionOffNoProjector(t *testing.T) {
	e, s, base := lookEnvE(t, func(e *testkit.Env) { e.Cat.Models[0].Projector = nil }, nil, false)
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	c, b := postLook(t, base, oneLook(pngB64(t, 8, 8), ""), nil)
	if c != 409 || codeOf(b) != "vision_off" {
		t.Fatalf("look: %d %s", c, b)
	}
	if upstreamBytes(t, e) != "" {
		t.Fatalf("the model server was asked: %s", upstreamBytes(t, e))
	}
	// a model that sleeps and has none: refused before it wakes
	e2, s2, base2 := lookEnvE(t, func(e *testkit.Env) { e.Cat.Models[0].Projector = nil }, func(c *run.Config) { c.StartSleeping = true }, false)
	testkit.WaitPhase(t, s2, "sleeping", 10*time.Second)
	c, b = postLook(t, base2, oneLook(pngB64(t, 8, 8), ""), nil)
	if c != 409 || codeOf(b) != "vision_off" || len(e2.Starts()) != 0 {
		t.Fatalf("sleeping, no projector: %d %s starts %d", c, b, len(e2.Starts()))
	}
}

// A sleeping engine that wants vision: the look wakes it, is held, and is answered once the model
// is up with its projector (the look's request is the one that wakes it).
func TestLookWakesASleepingModel(t *testing.T) {
	e, s, base := lookEnvE(t, nil, func(c *run.Config) { c.StartSleeping = true }, false)
	testkit.WaitPhase(t, s, "sleeping", 10*time.Second)
	if len(e.Starts()) != 0 {
		t.Fatal("a model server is running")
	}
	c, b := postLook(t, base, oneLook(pngB64(t, 40, 20), `,"detail":"low"`), nil)
	if c != 200 || !strings.Contains(b, "saw 1 images") {
		t.Fatalf("look on a sleeping model: %d %s", c, b)
	}
	if !s.VisionLoaded() || len(e.Starts()) == 0 {
		t.Fatal("answered without a model server with its projector")
	}
	_, parts := childOf(t, e)
	if len(parts) != 2 || parts[0].Type != "image_url" {
		t.Fatalf("parts %+v", parts)
	}
	// the same, streamed, after a sleep
	if err := s.Sleep(); err != nil {
		t.Fatal(err)
	}
	testkit.WaitPhase(t, s, "sleeping", 10*time.Second)
	c, b = postLook(t, base, oneLook(pngB64(t, 40, 20), `,"stream":true`), nil)
	if c != 200 || !strings.Contains(b, `"object":"look.result"`) || !strings.Contains(b, "data: [DONE]") {
		t.Fatalf("streamed look on a sleeping model: %d %s", c, b)
	}
}

// A sleeping engine that does not want vision: the look wakes it and then is refused vision_off
// (the model is up with no projector): the held look gets the refusal, as a JSON answer or, when
// streamed, in its stream.
func TestLookSleepingWithoutVisionRefusedAfterWake(t *testing.T) {
	e, s, base := lookEnvE(t, nil, func(c *run.Config) { c.StartSleeping = true; c.Vision = false }, false)
	testkit.WaitPhase(t, s, "sleeping", 10*time.Second)
	c, b := postLook(t, base, oneLook(pngB64(t, 8, 8), ""), nil)
	if c != 409 || codeOf(b) != "vision_off" {
		t.Fatalf("look: %d %s", c, b)
	}
	if upstreamBytes(t, e) != "" {
		t.Fatalf("the model server was asked: %s", upstreamBytes(t, e))
	}
	if err := s.Sleep(); err != nil {
		t.Fatal(err)
	}
	testkit.WaitPhase(t, s, "sleeping", 10*time.Second)
	c, b = postLook(t, base, oneLook(pngB64(t, 8, 8), `,"stream":true`), nil)
	if !strings.Contains(b, "vision") || strings.Contains(b, "look.result") {
		t.Fatalf("streamed look: %d %s", c, b)
	}
	if upstreamBytes(t, e) != "" {
		t.Fatalf("the model server was asked: %s", upstreamBytes(t, e))
	}
}

func gatedLookEnv(t *testing.T) (*testkit.Env, *run.Supervisor, string) {
	return lookEnvE(t, func(e *testkit.Env) {
		e.Llama["FAKELLAMA_STREAM_GATE"] = "1"
		e.Llama["FAKELLAMA_STREAM_DELAY"] = "1ms" // named moment: the fake's pace between tokens
	}, nil, true)
}

// A remote caller has one look at a time: a streamed look held in the model server (the fake
// sends its next chunk only when the test says) blocks that identity's next look with 429 busy,
// while another identity's look and a local look are admitted; after it ends the caller may look again.
func TestLookOneAtATimePerRemoteCaller(t *testing.T) {
	e, s, base := gatedLookEnv(t)
	_, rs, _ := remoteSrv(t, s)
	img := pngB64(t, 16, 16)
	req, _ := http.NewRequest("POST", rs.URL+"/engine/v1/look", strings.NewReader(oneLook(img, `,"stream":true`)))
	for k, v := range as(1) {
		req.Header.Set(k, v)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	br := bufio.NewReader(resp.Body)
	first, err := br.ReadString('\n') // the look is in flight: its first chunk is read
	if err != nil || !strings.Contains(first, "t0 ") {
		t.Fatalf("first chunk %q %v", first, err)
	}
	post := func(n int, base string) (int, string) {
		h := as(n)
		if n == 0 {
			h = nil
		}
		return postLook(t, base, oneLook(img, ""), h)
	}
	if c, b := post(1, rs.URL); c != 429 || codeOf(b) != "busy" {
		t.Fatalf("the same caller's second look: %d %s", c, b)
	}
	if c, b := post(2, rs.URL); c != 200 {
		t.Fatalf("another caller's look: %d %s", c, b)
	}
	if c, b := post(0, base); c != 200 {
		t.Fatalf("a local look: %d %s", c, b)
	}
	for i := 1; i <= 4; i++ {
		e.Touch(fmt.Sprintf("go.%d", i))
	}
	rest, _ := io.ReadAll(br)
	if !strings.Contains(string(rest), "look.result") {
		t.Fatalf("the held look's stream: %q", rest)
	}
	// the look has ended: its caller's slot in the gate is free (the release runs as the handler returns)
	testkit.WaitFor(t, 15*time.Second, "the caller may look again", func() bool {
		c, _ := post(1, rs.URL)
		return c == 200
	})
}

// The remote look's size: a look whose built request would pass maxRemoteBody (16 MiB) is
// look_too_large. The source is a 2800x2800 paletted PNG of random indices (7.8 MB, within the
// image limits) whose re-encoding as true colour is about three times as large.
func TestLookTooLargeOnceBuilt(t *testing.T) {
	e, s, _ := lookEnvE(t, func(e *testkit.Env) {
		e.Cat.Models[0].Projector.Detail = catalog.Detail{Low: 64, High: 4096, Max: 4096}
	}, nil, true)
	_, rs, _ := remoteSrv(t, s)
	rng := rand.New(rand.NewSource(1))
	pal := make(color.Palette, 256)
	for i := range pal {
		pal[i] = color.RGBA{uint8(rng.Intn(256)), uint8(rng.Intn(256)), uint8(rng.Intn(256)), 255}
	}
	const side = 2800
	src := image.NewPaletted(image.Rect(0, 0, side, side), pal)
	for i := range src.Pix {
		src.Pix[i] = uint8(rng.Intn(256))
	}
	var pb bytes.Buffer
	if err := png.Encode(&pb, src); err != nil {
		t.Fatal(err)
	}
	if pb.Len() > lookimg.Default.MaxBytes {
		t.Fatalf("the source is %d bytes: over the image limit", pb.Len())
	}
	body := oneLook(base64.StdEncoding.EncodeToString(pb.Bytes()), "")
	c, b := postLook(t, rs.URL, body, as(1))
	if c != 400 || codeOf(b) != "look_too_large" {
		t.Fatalf("remote look: %d %.300s", c, b)
	}
	if upstreamBytes(t, e) != "" {
		t.Fatal("the model server was asked")
	}
}

// A remote caller's look runs in its own slots: id_slot 0 lands on its first engine slot, and an
// id_slot outside its range is refused as for a chat request. A raw chat request of the same
// caller with an image is still refused, and the look with that image is not.
func TestLookRemoteSlotsAndMedia(t *testing.T) {
	e, s, _ := lookEnvE(t, func(e *testkit.Env) {
		e.Llama["FAKELLAMA_ID_SLOT"] = "1"
		e.Llama["FAKELLAMA_STREAM_DELAY"] = "1ms"
		e.Prof.CtxPerSlot.Max, e.Prof.CtxPerSlot.Floor = 8192, 4096
	}, func(c *run.Config) { c.ProfileGiven = true }, true)
	_, rs, _ := remoteSrv(t, s)
	A := growRemote(t, e, s, rs, 1, "a", 2) // client-1: engine slots 3, 4
	B := growRemote(t, e, s, rs, 2, "b", 1) // client-2: engine slot 5
	_, _, firstA := slotsOf(A.untilGiven(2))
	_, _, firstB := slotsOf(B.untilGiven(1))
	if firstA != 3 || firstB != 5 {
		t.Fatalf("ranges: a from %d, b from %d", firstA, firstB)
	}
	testkit.WaitFor(t, 20*time.Second, "vision loaded after the grows", func() bool { return s.VisionLoaded() })
	img := pngB64(t, 20, 20)
	for _, c := range []struct {
		n    int
		slot int
		want float64
	}{{1, 0, 3}, {1, 1, 4}, {2, 0, 5}} {
		code, body := postLook(t, rs.URL, oneLook(img, fmt.Sprintf(`,"id_slot":%d`, c.slot)), as(c.n))
		if code != 200 || upstreamSlot(t, e) != c.want {
			t.Fatalf("client %d id_slot %d: %d %s upstream slot %v", c.n, c.slot, code, body, upstreamSlot(t, e))
		}
	}
	before := upstreamBytes(t, e)
	for _, c := range []struct{ n, slot int }{{1, 2}, {2, 1}, {1, 1023}} {
		code, body := postLook(t, rs.URL, oneLook(img, fmt.Sprintf(`,"id_slot":%d`, c.slot)), as(c.n))
		if code != 400 || codeOf(body) != "slot_out_of_range" {
			t.Fatalf("client %d id_slot %d: %d %s", c.n, c.slot, code, body)
		}
	}
	if upstreamBytes(t, e) != before {
		t.Fatal("an out of range look reached the model server")
	}
	// no id_slot: the request goes to the caller's own slot
	if code, body := postLook(t, rs.URL, oneLook(img, ""), as(2)); code != 200 || upstreamSlot(t, e) != float64(5) {
		t.Fatalf("no id_slot: %d %s slot %v", code, body, upstreamSlot(t, e))
	}
	// the raw chat request with an image is refused for the same caller
	raw := fmt.Sprintf(`{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,%s"}},{"type":"text","text":"q"}]}]}`, img)
	code, body, _ := do(t, "POST", rs.URL+"/v1/chat/completions", raw, as(2))
	if code != 400 || codeOf(body) != "images_not_offered" {
		t.Fatalf("raw image chat: %d %s", code, body)
	}
	if code, body := postLook(t, rs.URL, oneLook(img, ""), as(2)); code != 200 || !strings.Contains(body, "saw 1 images") {
		t.Fatalf("look with the same image: %d %s", code, body)
	}
}

// A paired token may post a look (it is not a control route the scope forbids), and look is said in
// status, in hello (keyless too) and in a paired token's status.
func TestLookPairedAndStatus(t *testing.T) {
	pe, paired := scopeEnv(t)
	c, b, _ := do(t, "POST", pe.srv.URL+"/engine/v1/look", `{}`, map[string]string{"Authorization": "Bearer " + paired, "X-Easy-Engine": "1", "Content-Type": "application/json"})
	if notAllowed(c, b) || c != 400 || codeOf(b) != "bad_request" {
		t.Fatalf("paired look: %d %s", c, b)
	}
	for name, h := range map[string]map[string]string{"paired status": bearer(paired), "master status": bearer(pe.master)} {
		if m := getJSON(t, pe.srv.URL+"/engine/v1/status", h); string(m["look"]) != "true" {
			t.Errorf("%s look: %s", name, m["look"])
		}
	}
	if m := getJSON(t, pe.srv.URL+"/engine/v1/hello", nil); string(m["look"]) != "true" {
		t.Errorf("keyless hello look: %s", m["look"])
	}
	if m := getJSON(t, pe.srv.URL+"/engine/v1/hello", bearer(paired)); string(m["look"]) != "true" {
		t.Errorf("hello look: %s", m["look"])
	}
	_, _, base := lookEnv(t, nil)
	if m := getJSON(t, base+"/engine/v1/status", nil); string(m["look"]) != "true" {
		t.Errorf("local status look: %s", m["look"])
	}
	// without a key a look from another machine is refused at the gate, not served
	if c, b := postLook(t, pe.srv.URL, `{}`, nil); c != 401 && c != 403 {
		t.Fatalf("a keyless look: %d %s", c, b)
	}
}

// The model server receives only what the engine built: its keys, the image re-encoded and scaled to
// the detail's edge (never up), JPEG in and out, labels and the question as text parts.
func TestLookChildRequest(t *testing.T) {
	e, _, base := lookEnv(t, nil)
	src := pngB64(t, 400, 200)
	allowed := map[string]bool{"messages": true, "max_tokens": true, "stream": true, "stream_options": true, "id_slot": true, "chat_template_kwargs": true}
	for _, f := range catalog.SamplerFields {
		allowed[f] = true // what the sampler policy adds
	}
	for _, c := range []struct {
		detail string
		w, h   int
	}{{"low", 64, 32}, {"high", 128, 64}, {"max", 256, 128}, {"", 128, 64}} {
		extra := ""
		if c.detail != "" {
			extra = fmt.Sprintf(`,"detail":%q`, c.detail)
		}
		if code, body := postLook(t, base, oneLook(src, extra), nil); code != 200 {
			t.Fatalf("%s: %d %s", c.detail, code, body)
		}
		top, parts := childOf(t, e)
		for k := range top {
			if !allowed[k] {
				t.Errorf("%s: the model server got the key %q", c.detail, k)
			}
		}
		for _, k := range []string{"messages", "max_tokens", "stream", "chat_template_kwargs"} {
			if _, ok := top[k]; !ok {
				t.Errorf("%s: no %q in %s", c.detail, k, upstreamBytes(t, e))
			}
		}
		if string(top["stream"]) != "false" || top["stream_options"] != nil || top["id_slot"] != nil {
			t.Errorf("%s: stream %s stream_options %s id_slot %s", c.detail, top["stream"], top["stream_options"], top["id_slot"])
		}
		mime, w, h, raw := decodedPart(t, parts[0])
		orig, _ := base64.StdEncoding.DecodeString(src)
		if mime != "image/png" || w != c.w || h != c.h || bytes.Equal(raw, orig) {
			t.Errorf("%s: %s %dx%d (want image/png %dx%d, bytes re-encoded)", c.detail, mime, w, h, c.w, c.h)
		}
	}
	// never up: 40x20 at high is 40x20, and the bytes are still the engine's own
	small := pngWithText(t, 40, 20)
	if code, body := postLook(t, base, oneLook(small, `,"detail":"max"`), nil); code != 200 {
		t.Fatalf("%d %s", code, body)
	}
	_, parts := childOf(t, e)
	_, w, h, raw := decodedPart(t, parts[0])
	orig, _ := base64.StdEncoding.DecodeString(small)
	if w != 40 || h != 20 || bytes.Equal(raw, orig) || bytes.Contains(raw, []byte("secret")) {
		t.Errorf("small image %dx%d same bytes %v", w, h, bytes.Equal(raw, orig))
	}
	// a JPEG comes out a JPEG; max_tokens, id_slot and stream_options in a streamed look
	jp := jpegB64(t, 300, 150)
	if code, body := postLook(t, base, oneLook(jp, `,"max_tokens":9,"id_slot":0,"stream":true`), nil); code != 200 {
		t.Fatalf("%d %s", code, body)
	}
	top, parts := childOf(t, e)
	mime, w, h, _ := decodedPart(t, parts[0])
	if mime != "image/jpeg" || w != 128 || h != 64 {
		t.Errorf("jpeg: %s %dx%d", mime, w, h)
	}
	if string(top["stream"]) != "true" || string(top["stream_options"]) != `{"include_usage":true}` || string(top["max_tokens"]) != "9" || string(top["id_slot"]) != "0" {
		t.Errorf("streamed look child: %s", upstreamBytes(t, e))
	}
	// two images with a label on the second: the text parts sit in order
	body := fmt.Sprintf(`{"images":[{"data":%q},{"data":%q,"label":"second"}],"question":"which?"}`, small, jp)
	if code, b := postLook(t, base, body, nil); code != 200 || !strings.Contains(b, "saw 2 images") {
		t.Fatalf("two images: %d %s", code, b)
	}
	_, parts = childOf(t, e)
	var order []string
	for _, p := range parts {
		if p.Type == "text" {
			order = append(order, "text:"+p.Text)
		} else {
			order = append(order, p.Type)
		}
	}
	if got := strings.Join(order, ","); got != "image_url,text:second,image_url,text:which?" {
		t.Errorf("parts %s", got)
	}
}

// The streamed look: chunks come as the model server sent them, then exactly one look.result just
// before [DONE] with the joined text and the usage chunk's figures.
func TestLookStreamEndToEnd(t *testing.T) {
	_, _, base := lookEnv(t, func(c *run.Config) {})
	req, _ := http.NewRequest("POST", base+"/engine/v1/look", strings.NewReader(oneLook(pngB64(t, 20, 20), `,"stream":true`)))
	req.Header.Set("X-Easy-Engine", "1")
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if ct := resp.Header.Get("Content-Type"); !strings.HasPrefix(ct, "text/event-stream") {
		t.Fatalf("content type %q", ct)
	}
	raw, _ := io.ReadAll(resp.Body)
	var events []string
	for _, ev := range strings.Split(strings.TrimSpace(string(raw)), "\n\n") {
		events = append(events, ev)
	}
	n := len(events)
	if n < 8 {
		t.Fatalf("events: %q", events)
	}
	for i := 0; i < 5; i++ {
		if want := fmt.Sprintf("data: {\"choices\":[{\"delta\":{\"content\":\"t%d \"}}]}", i); events[i] != want {
			t.Fatalf("chunk %d: %q (want the model server's own %q)", i, events[i], want)
		}
	}
	if events[n-1] != "data: [DONE]" {
		t.Fatalf("last event %q", events[n-1])
	}
	var res struct {
		Object string
		Text   string
		Tokens map[string]int
		MS     map[string]int
	}
	if !strings.HasPrefix(events[n-2], "data: ") || json.Unmarshal([]byte(strings.TrimPrefix(events[n-2], "data: ")), &res) != nil || res.Object != "look.result" {
		t.Fatalf("the event before [DONE]: %q", events[n-2])
	}
	if res.Text != "t0 t1 t2 t3 t4 " || res.Tokens["image"] != -1 || res.Tokens["prompt"] < 100 || res.Tokens["completion"] != 5 || res.MS["prefill"] != 13 || res.MS["total"] != 113 {
		t.Fatalf("look.result %+v", res)
	}
	if c := strings.Count(string(raw), "look.result"); c != 1 {
		t.Fatalf("%d look.result events", c)
	}
	// the usage chunk is among the chunks passed through
	if !strings.Contains(string(raw), `"usage"`) {
		t.Fatalf("no usage chunk passed through: %q", raw)
	}
}
