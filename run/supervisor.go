package run

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/CharlesMod/easyEngine/catalog"
	"github.com/CharlesMod/easyEngine/core/fit"
	"github.com/CharlesMod/easyEngine/event"
	"github.com/CharlesMod/easyEngine/lifecycle"
	"github.com/CharlesMod/easyEngine/plan"
	"github.com/CharlesMod/easyEngine/probe"
	"github.com/CharlesMod/easyEngine/probe/gpusense"
)

// ErrBusy means an install job already runs.
var ErrBusy = errors.New("an install job is already running")

// Progress is the download progress shown in the status.
type Progress struct {
	DoneBytes  int64   `json:"done_bytes"`
	TotalBytes int64   `json:"total_bytes"`
	RateBps    float64 `json:"rate_bps"`
	ETAS       int     `json:"eta_s"`
	Step       string  `json:"step"`
}

// ShapeInfo is the running shape.
type ShapeInfo struct {
	Slots      int    `json:"slots"`
	CtxPerSlot int    `json:"ctx_per_slot"`
	NCpuMoe    int    `json:"n_cpu_moe"`
	KVType     string `json:"kv_type"`
	VRAMMiB    int    `json:"vram_mib"`
	// VRAMBasis says whether VRAMMiB was read from the running server (measured) or is the
	// planner's figure (predicted).
	VRAMBasis fit.Basis `json:"vram_basis"`
}

// ExternalInfo is the external endpoint without its key.
type ExternalInfo struct {
	BaseURL   string `json:"base_url"`
	Model     string `json:"model"`
	APIKeySet bool   `json:"api_key_set"`
}

// Status is GET /engine/v1/status.
type Status struct {
	Version string `json:"version"`
	// API is the version of the control API (APIVersion).
	API      int       `json:"api"`
	Port     int       `json:"port"`
	Mode     string    `json:"mode"`
	Phase    string    `json:"phase"`
	Message  string    `json:"message"`
	Progress *Progress `json:"progress"`
	Model    string    `json:"model"`
	Backend  string    `json:"backend"`
	// Build is the llama.cpp build: its release tag, patched or stock, its rung and why, the
	// pin and the move to it (build.go); nil with no build installed.
	Build     *BuildInfo   `json:"build,omitempty"`
	Shape     *ShapeInfo   `json:"shape"`
	ServerURL string       `json:"server_url"`
	Agent     AgentStatus  `json:"agent"`
	Parents   []Parent     `json:"parents"`
	TokS      *float64     `json:"tok_s"`
	Errors    []string     `json:"errors"`
	External  ExternalInfo `json:"external"`
	LingerS   int          `json:"linger_s"`
	// First-run staging (docs/ENGINE.md section 7a).
	ModelTier  string       `json:"model_tier"`
	ModelStage string       `json:"model_stage"`
	Upgrade    *UpgradeInfo `json:"upgrade"`
	KVType     string       `json:"kv_type"`
	Restarts   int          `json:"restarts"`
	// ModelFallback is true when the selected model is a catalog fallback (the smallest one);
	// ModelNote says what to expect from it. Front ends show it as a warning.
	ModelFallback bool   `json:"model_fallback"`
	ModelNote     string `json:"model_note,omitempty"`
	// Sampler is the sampler policy mode and the profiles in force (docs/SAMPLERS.md).
	Sampler SamplerInfo `json:"sampler"`
	// Contexts is the prompt cache kept across a reshape (run/context.go): the slots saved while
	// no model server holds them; nil when none.
	Contexts *ContextsInfo `json:"contexts,omitempty"`
	// ModelName is the human name of the selected model; Custom is the user's own GGUF file
	// (nil when none is set). While it serves, Model is "custom:<name>".
	ModelName string      `json:"model_name"`
	Custom    *CustomInfo `json:"custom"`
	// ModelParams is the total parameter count of the served custom file (0 unknown);
	// ModelSizeClass is "small" (< 8e9), "large" or "" (custom.go).
	ModelParams    int64  `json:"model_params"`
	ModelSizeClass string `json:"model_size_class"`
	// Previous is the model the last switch replaced; its file is still on disk (run/switch.go).
	Previous *PreviousInfo `json:"previous"`
	// GPU is the adapters and the measured reserve as the watchdog last read them (null before
	// the first reading); GET /engine/v1/gpu has the processes too.
	GPU *gpusense.GPUSummary `json:"gpu"`
	// Device is what llama-server was given to hold it to the planned adapter (null before the
	// first start): docs/ENGINE.md section 4a.
	Device *plan.DevicePin `json:"device"`
	// Recovery is the crash and failed-start record while the engine recovers (null when there
	// is none): run/recover.go, docs/ENGINE.md section 5.
	Recovery *RecoveryInfo `json:"recovery"`
	// Yield is set while the card is yielded to another program (phase "yielded"), or while
	// replies drain before that: run/yield.go, docs/ENGINE.md section 6b.
	Yield *YieldInfo `json:"yield"`
	// Held is the count of requests the door holds until the model is up (Supervisor.Hold).
	Held int `json:"held"`
	// Attachments are the clients' attachments (the base first), Wants the merged wants with who
	// is unserved and the vacant ranges of the running mind (run/clients.go).
	Attachments []AttachView `json:"attachments"`
	Wants       WantsInfo    `json:"wants"`
	// Vision is true while the running server has the vision projector loaded; VisionNote says
	// why not when vision is wanted ("" otherwise); VisionProgress is the projector's download
	// (null when none runs). run/vision.go, docs/ENGINE.md "Vision".
	Vision         bool      `json:"vision"`
	VisionNote     string    `json:"vision_note"`
	VisionProgress *Progress `json:"vision_progress"`
}

// Parent is a watched parent process.
type Parent struct {
	Pid   int  `json:"pid"`
	Alive bool `json:"alive"`
}

// Settings is PUT /engine/v1/settings.
type Settings struct {
	Mode     string   `json:"mode"`
	External External `json:"external"`
	// ClearAPIKey removes the stored key (an empty api_key keeps it).
	ClearAPIKey bool   `json:"clear_api_key,omitempty"`
	LingerS     int    `json:"linger_s"`
	KVType      string `json:"kv_type"`
	// SamplerPolicy is enforce | fill | off; empty keeps the stored value.
	SamplerPolicy string `json:"sampler_policy,omitempty"`
	// Streamer is the token streamer, on | off (docs/STREAMING.md); empty keeps the stored value.
	Streamer string `json:"streamer,omitempty"`
	// KeepContext is on | off | always; empty keeps the stored value.
	KeepContext string `json:"keep_context,omitempty"`
	// Vision loads the served model's vision projector (run/vision.go); nil keeps the stored value.
	Vision *bool `json:"vision,omitempty"`
	// VisionForced is read-only (GET): serve --vision holds vision on for this run, so the stored
	// Vision cannot turn it off. Absent when not forced.
	VisionForced bool `json:"vision_forced,omitempty"`
}

// VisionForcedMsg is the words of the refusal to turn vision off while serve --vision holds it on.
const VisionForcedMsg = "vision is on for this run by serve --vision; the setting cannot turn it off until the engine runs without it"

// CheckVisionOff refuses (409 vision_forced) a request to turn vision off while serve --vision
// holds it on, whatever the stored setting is: a PUT that says "vision": false must not look as if
// it worked. nil asks nothing about vision. Nothing is changed.
func (s *Supervisor) CheckVisionOff(v *bool) *ClientError {
	if v != nil && !*v && s.cfg.Vision {
		return cerr(409, "vision_forced", "%s", VisionForcedMsg)
	}
	return nil
}

// Route tells the proxy where a request goes.
type Route struct {
	Kind    string // local | external | unavailable
	Code    string // for unavailable: "yielded" while the card is yielded, else empty
	URL     string // base URL (no trailing slash); for external it includes any /v1
	Key     string
	Message string
	// Hold is set on an unavailable route while the model is on its way up (starting, loading,
	// resizing, the swap to the full model, a sleep a request wakes): the door holds the request
	// (Supervisor.Hold) and enters again when the phase changes, instead of answering 503.
	Hold bool
}

type server struct {
	proc      *lifecycle.Proc
	port      int
	shape     fit.Shape
	key       string
	build     string // the build it runs (install.build): a failure is recorded under it (recordLC)
	fp        fit.Footprint
	seed      fit.Footprint // the header's seed, which every reading is held to (never fp)
	startedAt time.Time
	nextGrow  time.Time
	measured  int
	reserve0  int         // the reserve the shape was chosen under (run/recover.go reads a crash against it)
	projMiB   int         // the vision projector's load (catalog load_mib) when it runs with one
	oom       *oomWatch   // out-of-memory lines in its output, counted by request (run/recover.go)
	drain     *drain      // a shrink waiting for the replies in flight
	planned   atomic.Bool // stopped on purpose: its end is not a crash
	// ctxKey is what a saved slot must have been saved under to be laid back into this server
	// (run/context.go): model file, llama build, cache type.
	ctxKey ctxKey
	// laidBack is the tokens each slot was given back at the start (GET /slots says nothing of
	// a restored slot until it serves). Guarded by opMu.
	laidBack map[int]int
	// oomShrink: the shrink ahead is for out-of-memory while serving; it saves nothing. Guarded by opMu.
	oomShrink bool
	// oomPend: out-of-memory lines in two requests were said of this server and not yet acted on
	// (they came mid-drain or mid-swap); oomLine is the last. Guarded by opMu.
	oomPend bool
	oomLine string
	// growWait cancels the named moment GrowBackoff's look-again (one at a time; mindDown cancels).
	growWait func() bool
	// stableCancel cancels the named moment CrashQuick after the start that clears the crash
	// record (run/recover.go).
	stableCancel func() bool
}

// Supervisor owns the children and the state.
type Supervisor struct {
	cfg  Config
	dir  string
	root context.Context
	stop context.CancelFunc

	log *lifecycle.LogSink

	mu          sync.Mutex
	st          State
	phase       string
	message     string
	progress    *Progress
	shape       *ShapeInfo
	device      *plan.DevicePin // what the last llama-server start was given (resolvePin)
	errs        []string
	port        int
	llamaURL    string
	userStopped bool
	jobRunning  bool
	jobCancel   context.CancelFunc
	loadCancel  context.CancelFunc
	waking      bool
	restarts    int
	hw          *probe.Hardware
	cat         *catalog.Catalog
	retryable   bool
	rec         recovery
	reshape     bool
	reshapeWhy  string      // what the queued reshape is for (the log line of its reload)
	vis         visionState // the vision projector (run/vision.go)
	tokS        float64

	// hub is what the supervisor says (run/book.go); poke wakes the mind's reactor and door is
	// the count of requests in flight with what it arms (run/mind.go); card is the room, the
	// reserve as sensed when needed (run/room.go).
	hub  *event.Hub
	poke chan struct{}
	// loadAsk is a request held while the model loads: the load asks /health once on it (Hold).
	loadAsk chan struct{}
	heldMu  sync.Mutex // the held count and its door fact (run/hold.go)
	// listeners counts the listeners for a word (listenWord): a request is said as a nudge only while one listens
	listeners atomic.Int32
	door      door
	kept      keptContexts // the saved slots (run/context.go)
	card      *roomWatch
	holders   *holderScan  // who holds the card's nodes, scanned once per word for the room and the yield
	sf        settingsFile // settings.json as the engine follows it (run/settings.go)
	// deferred: started asleep (Config.StartSleeping); the agent and an unfinished upgrade wait
	// for the first Start.
	deferred atomic.Bool

	upg       *upgrade // the starter-to-target upgrade, if this run has one (starter.go)
	upgCancel context.CancelFunc
	// move is the install of the catalog's pin beside the build that runs (build.go); nil when
	// none. moveCancel is set while its job runs. vmm is the patched server's CUDA buffer line.
	move       *BuildNext
	moveCancel context.CancelFunc
	vmm        string
	buildMu    sync.Mutex // one build install at a time: the job's and the move's may name one build
	// smoking is the pid of a smoke test the engine runs (install.go smokeOwned): the engine's
	// own for the room watch. smokeFail is the staged build whose smoke test failed at a load,
	// with what stood then; the same tag is not tried again until one of them changed.
	smoking   ourPID
	smokeFail *smokeFailure
	// removeGen counts the removes of the downloads (under mu): a move that began before one does
	// not stage what it fetched. words counts the words a failed move waits for (moveWord).
	removeGen uint64
	words     atomic.Uint64
	swapping  bool
	removing  bool          // remove-downloads job running (storage.go)
	stoCache  *storageCache // last complete storage walk

	opMu sync.Mutex
	srv  *server
	ours ourPID // srv's pid, for readers that do not hold opMu
	// launching is the llama-server pid from its launch on (ours is set only once healthy);
	// yieldInfo is set while the card is yielded (run/yield.go). yieldInfo is under mu.
	launching ourPID
	yieldInfo *YieldInfo
	gpu       gpuState

	agent agentMgr
	// clients is the attachments and the running mind's slot map (run/clients.go).
	clients clientReg
	wg      sync.WaitGroup
}

// New builds a supervisor. Call Run to start the watchdog.
func New(cfg Config) (*Supervisor, error) {
	cfg.defaults()
	ownReads(&cfg)
	if cfg.DataDir == "" {
		return nil, errors.New("no data dir")
	}
	for _, d := range []string{"", "logs", "backends", "models", "runtime", "downloads"} {
		if err := os.MkdirAll(filepath.Join(cfg.DataDir, d), 0o755); err != nil {
			return nil, err
		}
	}
	lg, _ := lifecycle.OpenLog(filepath.Join(cfg.DataDir, "logs", "engine.log"))
	ctx, cancel := context.WithCancel(context.Background())
	st, file, note, serr := loadAll(cfg.DataDir)
	hub, err := event.New(book, cfg.Now)
	if err != nil {
		cancel()
		return nil, err
	}
	s := &Supervisor{cfg: cfg, dir: cfg.DataDir, root: ctx, stop: cancel, log: lg, st: st, hub: hub, poke: make(chan struct{}, 1), loadAsk: make(chan struct{}, 1)}
	s.sf.begin(file, serr)
	s.vis.setting = file.Vision
	s.agent.s = s
	s.holders = &holderScan{s: s}
	s.card = newRoomWatch(s)
	s.say("engine", "", map[string]any{"version": cfg.Version, "api": APIVersion, "port": 0})
	s.say("door", "", map[string]int{"held": 0})
	s.sayModel()
	if note != "" {
		s.addError("%s", note)
	}
	s.setPhaseLocked("not_installed", "No model installed")
	s.contextsInit()
	s.clientsInit()
	s.sayClients()
	return s, nil
}

// SetPort records the control API port (the agent is pointed at it).
func (s *Supervisor) SetPort(p int) {
	s.mu.Lock()
	s.port = p
	s.mu.Unlock()
	s.say("engine", "", map[string]any{"version": s.cfg.Version, "api": APIVersion, "port": p})
}

func (s *Supervisor) logf(format string, a ...any) {
	if s.log == nil {
		return
	}
	fmt.Fprintf(s.log, "%s %s\n", s.cfg.Now().Format("2006-01-02T15:04:05"), fmt.Sprintf(format, a...))
}

func (s *Supervisor) addError(format string, a ...any) {
	msg := fmt.Sprintf(format, a...)
	s.logf("error: %s", msg)
	s.mu.Lock()
	s.errs = append(s.errs, msg)
	if len(s.errs) > 10 {
		s.errs = s.errs[len(s.errs)-10:]
	}
	s.mu.Unlock()
	s.say("error", "", map[string]string{"message": msg})
}

func (s *Supervisor) setPhaseLocked(phase, msg string) {
	s.phase, s.message = phase, msg
	s.say("phase", "", map[string]string{"phase": phase, "message": msg})
	s.refreshLocked()
	if phase == "ready" { // the reactor looks once: what waited for ready (an out-of-memory) is acted on
		select {
		case s.poke <- struct{}{}:
		default:
		}
	}
}

// refreshLocked says the facts that status computes from several fields: the shape, the device,
// the yield, the upgrade, the recovery record and the job. The hub says only what changed (an
// identical body is said already). Caller holds s.mu.
func (s *Supervisor) refreshLocked() {
	if s.hub == nil {
		return
	}
	sayPtr(s, "mind.shape", "", s.shape)
	sayPtr(s, "mind.device", "", s.device)
	sayPtr(s, "yield", "", s.yieldInfo)
	var st Status
	s.stagingLocked(&st)
	sayPtr(s, "upgrade", "", st.Upgrade)
	sayPtr(s, "build", "", s.buildInfoLocked())
	sayPtr(s, "recovery", "", s.recoveryLocked())
	s.say("vision", "", s.visionInfoLocked())
	switch {
	case s.removing:
		s.say("job", "", map[string]any{"kind": "remove", "running": true, "install": s.jobRunning})
	case s.jobRunning:
		s.say("job", "", map[string]any{"kind": "install", "running": true})
	default:
		s.sayGone("job", "")
	}
}

// refresh is refreshLocked for a caller that does not hold s.mu.
func (s *Supervisor) refresh() {
	s.mu.Lock()
	s.refreshLocked()
	s.mu.Unlock()
}

// sayModel says the selected model and how it runs.
func (s *Supervisor) sayModel() {
	s.mu.Lock()
	b := map[string]any{"mode": s.st.Mode, "model": s.st.Model, "backend": s.st.Backend, "kv_type": s.st.KVType,
		"target": s.st.Target, "previous": s.st.Previous, "starter_owned": s.st.StarterOwned, "custom": s.st.Custom,
		"sampler_policy": s.st.SamplerPolicy, "linger_s": s.st.LingerS}
	s.mu.Unlock()
	s.say("model", "", b)
}

func (s *Supervisor) setPhase(phase, msg string) {
	s.mu.Lock()
	s.setPhaseLocked(phase, msg)
	s.mu.Unlock()
	s.logf("phase %s: %s", phase, msg)
}

// Phase returns the current phase.
func (s *Supervisor) Phase() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.phase
}

// Status returns a snapshot.
func (s *Supervisor) Status() Status {
	atts, wi := s.attachmentsInfo(), s.wantsInfo()
	s.mu.Lock()
	defer s.mu.Unlock()
	st := Status{
		Version: s.cfg.Version, API: APIVersion, Port: s.port, Mode: s.st.Mode, Phase: s.phase, Message: s.message,
		Model: s.st.Model, Backend: s.st.Backend, Errors: append(append([]string{}, s.errs...), s.sf.notes()...),
		LingerS: s.st.LingerS, KVType: s.st.KVType, Restarts: s.restarts,
		External: ExternalInfo{BaseURL: s.st.External.BaseURL, Model: s.st.External.Model, APIKeySet: s.st.External.APIKey != ""},
		Parents:  []Parent{},
	}
	if s.progress != nil {
		p := *s.progress
		st.Progress = &p
	}
	if s.shape != nil {
		sh := *s.shape
		st.Shape = &sh
	}
	if s.phase == "ready" || s.phase == "resizing" {
		st.ServerURL = s.llamaURL
	}
	if s.cfg.Parents != nil {
		for _, p := range s.cfg.Parents() {
			st.Parents = append(st.Parents, Parent{Pid: p, Alive: lifecycle.PidAlive(p)})
		}
	}
	if s.tokS > 0 && (s.phase == "ready" || s.phase == "resizing") {
		v := s.tokS
		st.TokS = &v
	}
	st.Agent = s.agent.status()
	if m, ok := s.selectedModelLocked(); ok && m.Fallback {
		st.ModelFallback, st.ModelNote = true, m.FallbackNote
	}
	s.stagingLocked(&st)
	st.Sampler = s.samplerLocked()
	s.customStatusLocked(&st)
	st.Previous = s.previousLocked()
	if s.yieldInfo != nil {
		y := *s.yieldInfo
		st.Yield = &y
	}
	if s.gpu.sum != nil {
		g := *s.gpu.sum
		st.GPU = &g
	}
	if s.device != nil {
		d := *s.device
		st.Device = &d
	}
	st.Recovery = s.recoveryLocked()
	vi := s.visionInfoLocked()
	st.Vision, st.VisionNote, st.VisionProgress = vi.Vision, vi.Note, s.visionProgressLocked()
	st.Held = int(s.door.held.Load())
	st.Contexts = s.kept.info()
	st.Attachments, st.Wants = atts, wi
	st.Build = s.buildInfoLocked()
	return st
}

// selectedModelLocked is the catalog entry of the selected model, if there is one. Caller holds
// s.mu. In external mode the selected local model is not in use, so it reports none.
func (s *Supervisor) selectedModelLocked() (catalog.Model, bool) {
	if s.st.Model == "" || s.st.Mode == "external" {
		return catalog.Model{}, false
	}
	c := s.cat
	if c == nil {
		c = s.cfg.Catalog
	}
	if c == nil {
		var err error
		if c, err = catalog.Load(s.dir); err != nil {
			return catalog.Model{}, false
		}
		s.cat = c
	}
	return c.Model(s.st.Model)
}

func (s *Supervisor) save() {
	s.mu.Lock()
	st := s.st
	st.Book.Entries = copyBook(s.st.Book.Entries)
	s.mu.Unlock()
	s.sayModel()
	s.wakeMind(func(*causes) {}) // a settings change (the KV type) is looked at by the reactor
	if err := writeState(s.dir, st); err != nil {
		s.logf("write state: %v", err)
	}
}

func copyBook(m map[string]fit.Calib) map[string]fit.Calib {
	c := make(map[string]fit.Calib, len(m))
	for k, v := range m {
		c[k] = v
	}
	return c
}

// ---- catalog, hardware, plan ------------------------------------------------------------

func (s *Supervisor) catalog() (*catalog.Catalog, error) {
	s.mu.Lock()
	c := s.cat
	s.mu.Unlock()
	if c != nil {
		return c, nil
	}
	c = s.cfg.Catalog
	if c == nil {
		var err error
		if c, err = catalog.Load(s.dir); err != nil {
			return nil, err
		}
	}
	s.mu.Lock()
	s.cat = c
	s.mu.Unlock()
	return c, nil
}

// Hardware returns the cached probe, or a fresh one.
func (s *Supervisor) Hardware(ctx context.Context, refresh bool) (probe.Hardware, error) {
	s.mu.Lock()
	h := s.hw
	s.mu.Unlock()
	if h != nil && !refresh {
		return *h, nil
	}
	hw, err := s.cfg.Probe(ctx)
	if err != nil {
		return hw, err
	}
	if st, err := diskFree(s.dir); err == nil && hw.DiskFreeMiB == 0 {
		hw.DiskFreeMiB = st
	}
	s.mu.Lock()
	s.hw = &hw
	s.mu.Unlock()
	return hw, nil
}

// wants is the engine's wants: the clients' merged (run/clients.go), or the profile's when no
// client states any.
func (s *Supervisor) wants() fit.Wants {
	w, _ := s.mergedWants()
	s.mu.Lock()
	w.KVType = s.st.KVType
	s.mu.Unlock()
	return w
}

// wantsFor applies the stored ctx cap of (model, device) to the profile's wants.
func (s *Supervisor) wantsFor(key string) fit.Wants {
	w := s.wants()
	s.mu.Lock()
	c := s.st.Book.Entries[key].CapCtxTotal
	s.mu.Unlock()
	if c > 0 && w.Slots > 0 {
		m := c / w.Slots / 1024 * 1024
		if m < w.CtxFloor {
			m = w.CtxFloor
		}
		if m < w.CtxMax {
			w.CtxMax = m
		}
	}
	return w
}

// Plan is GET /engine/v1/plan.
func (s *Supervisor) Plan(ctx context.Context) (plan.PlanResult, error) {
	hw, err := s.Hardware(ctx, false)
	if err != nil {
		return plan.PlanResult{}, err
	}
	cat, err := s.catalog()
	if err != nil {
		return plan.PlanResult{}, err
	}
	sn := s.measure(ctx, hw, s.ours.pids())
	need := func(m catalog.Model) plan.DiskNeed { return s.setupNeed(cat, hw, "", m) }
	res := plan.PlanSensedDisk(cat, s.liveDisk(hw), s.wants(), sn.sum, s.modelInstalled, nil, need)
	res.GPU = sn.report()
	s.mu.Lock()
	running, be := s.device, s.st.Backend
	if s.shape == nil { // no server up
		running = nil
	}
	s.mu.Unlock()
	switch {
	case running != nil: // what the server in use was given
		d := *running
		res.Device = &d
	case be != "": // the installed build's, not the predicted one's
		d := plan.PinDevice(sn.sum, plan.AccelOf(be), nil)
		res.Device = &d
	}
	s.planCustom(&res, cat, hw, sn)
	s.fillDisk(&res, cat)
	return res, nil
}

func (s *Supervisor) modelPath(m catalog.Model) string { return filepath.Join(s.dir, "models", m.File) }

func (s *Supervisor) modelInstalled(m catalog.Model) bool {
	st, err := os.Stat(s.modelPath(m))
	return err == nil && st.Size() > 0 && (m.Size == 0 || st.Size() == m.Size)
}

func serverBinName() string {
	if runtime.GOOS == "windows" {
		return "llama-server.exe"
	}
	return "llama-server"
}

// findServer locates llama-server inside an unpacked backend (archives nest it).
func findServer(dir string) string {
	var found string
	filepath.WalkDir(dir, func(p string, d os.DirEntry, err error) error {
		if err == nil && !d.IsDir() && d.Name() == serverBinName() && found == "" {
			found = p
		}
		return nil
	})
	return found
}

type install struct {
	model     catalog.Model
	modelPath string
	bin       string
	backend   string
	tag       string // the llama.cpp release the build is from; "" when installed before tags were recorded
	// build is the engine build a cap is measured with and the prompt cache is keyed by
	// (buildKey): the backend id, the release's full tag and the sha256 of the archive installed
	// (its .installed marker), so b12651 and b12651-easy1 are two builds.
	build string
}

func (s *Supervisor) resolveInstall() (install, bool) {
	s.mu.Lock()
	id, be, tag := s.st.Model, s.st.Backend, s.st.Build
	s.mu.Unlock()
	if id == "" || be == "" {
		return install{}, false
	}
	cat, err := s.catalog()
	if err != nil {
		return install{}, false
	}
	var m catalog.Model
	var mpath string
	if id == customID {
		cm, ok := s.customModel(cat)
		if !ok {
			return install{}, false
		}
		if st, err := os.Stat(cm.File); err != nil || st.IsDir() {
			return install{}, false
		}
		m, mpath = cm, cm.File
	} else {
		var ok bool
		if m, ok = cat.Model(id); !ok || !s.modelInstalled(m) {
			return install{}, false
		}
		mpath = s.modelPath(m)
	}
	dir := backendDir(s.dir, be, tag)
	bin := findServer(dir)
	if bin == "" {
		return install{}, false
	}
	sha := ""
	if b, err := os.ReadFile(filepath.Join(dir, ".installed")); err == nil {
		sha = strings.TrimSpace(string(b))
	}
	return install{model: m, modelPath: mpath, bin: bin, backend: be, tag: tag, build: buildKey(be, tag, sha)}, true
}

// ---- proxy helpers ------------------------------------------------------------------------

// Route says where the proxy should send a request.
func (s *Supervisor) Route() Route {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.routeLocked()
}

// routeLocked is Route; the caller holds s.mu. A route with Hold set is one the door holds the
// request for (docs/ENGINE.md, "The door holds"): the model is on its way up.
func (s *Supervisor) routeLocked() Route {
	if s.st.Mode == "external" {
		return Route{Kind: "external", URL: strings.TrimRight(s.st.External.BaseURL, "/"), Key: s.st.External.APIKey}
	}
	if y := s.yieldInfo; y != nil {
		return Route{Kind: "unavailable", Code: "yielded", Message: "Paused while " + y.Reason + " runs"}
	}
	if s.phase == "error" {
		// a request at the door is a word that could change the outcome (run/mind.go)
		s.nudge("a request at the door")
	}
	switch s.phase {
	case "ready":
		if s.swapping { // a swap has begun and its phase is still to be said
			return Route{Kind: "unavailable", Hold: true, Message: "Switching to the full model"}
		}
		return Route{Kind: "local", URL: s.llamaURL}
	case "sleeping":
		if s.userStopped { // a person's Stop: a request does not wake it
			return Route{Kind: "unavailable", Message: "Loading model"}
		}
		if !s.waking {
			s.waking = true
			go s.wake()
		}
		return Route{Kind: "unavailable", Hold: true, Message: "Loading model"}
	case "not_installed":
		return Route{Kind: "unavailable", Message: "No model installed"}
	case "resizing":
		if s.swapping {
			return Route{Kind: "unavailable", Hold: true, Message: "Switching to the full model"}
		}
		return Route{Kind: "unavailable", Hold: true, Message: "Loading model"}
	case "starting", "loading":
		return Route{Kind: "unavailable", Hold: true, Message: "Loading model"}
	case "stopping":
		return Route{Kind: "unavailable", Message: "The engine is stopping"}
	case "error":
		return Route{Kind: "unavailable", Message: "Model server error: " + s.message}
	}
	return Route{Kind: "unavailable", Message: "Loading model"}
}

func (s *Supervisor) setProgress(step string, p *Progress) {
	s.mu.Lock()
	if p != nil {
		p.Step = step
	}
	s.progress = p
	sayPtr(s, "install.progress", "", p)
	s.mu.Unlock()
}

// ---- lifecycle -----------------------------------------------------------------------------

// Run starts the watchdog and, in local mode with an installed model, the server.
// It returns at once; Shutdown ends everything.
func (s *Supervisor) Run() {
	s.settle()
	if !s.cfg.NoAutoStart {
		s.mu.Lock()
		local := s.st.Mode == "local"
		s.mu.Unlock()
		// An agent the profile has and the data dir lacks is installed first, by itself: in
		// external mode (no model install ever runs there), and beside a model that is installed
		// (its agent was removed). With no model yet, the model's install job installs it.
		needAgent := s.agentWanted() && !s.agentReady()
		if needAgent && local {
			_, needAgent = s.resolveInstall()
		}
		if s.cfg.StartSleeping {
			// the app waits for a connection: no model is loaded, and running the agent and an
			// unfinished upgrade wait for the first start too (Start runs them once); a missing
			// agent is still installed now
			s.deferred.Store(true)
			s.mu.Lock()
			asleep := s.phase == "starting" // installed, local, not stopped (settle)
			s.mu.Unlock()
			if asleep {
				s.setPhase("sleeping", "Model not loaded")
			}
			if needAgent {
				s.wg.Add(1)
				go func() {
					defer s.wg.Done()
					s.installAgentThen(func() {
						if !s.deferred.Load() { // the first start came while it installed: run it now
							s.startAgent()
						}
					})
				}()
			}
		} else {
			s.startNow(local, needAgent)
		}
	}
	s.wg.Add(1)
	go func() { defer s.wg.Done(); s.mind() }()
	s.watchNetwork()
	if !s.cfg.NoAutoStart {
		s.checkPin() // the engine's start: a build other than the catalog's pin moves to it
	}
	if s.cfg.Yield != nil {
		s.wg.Add(1)
		go func() { defer s.wg.Done(); s.yieldLoop() }()
	}
}

// startNow is Run's start when the engine does not start asleep: the agent (installed first when
// it is missing), the model in local mode, and an unfinished upgrade.
func (s *Supervisor) startNow(local, needAgent bool) {
	if !needAgent {
		s.startAgent()
	}
	if local || needAgent {
		s.wg.Add(1)
		go func() {
			defer s.wg.Done()
			start := func() {
				if local {
					s.Start()
				}
			}
			if needAgent { // the first attempt comes first; a failed one does not hold the model back
				s.installAgentThen(func() { s.startAgent(); start() })
				return
			}
			start()
		}()
	}
	if local {
		s.resumeUpgrade()
	}
}

// settle picks the resting phase when nothing is running.
func (s *Supervisor) settle() {
	s.mu.Lock()
	ext := s.st.Mode == "external"
	stopped := s.userStopped
	s.mu.Unlock()
	if ext {
		s.setPhase("external", "Using your own endpoint")
		return
	}
	if _, ok := s.resolveInstall(); !ok {
		if !s.customMissingPhase() {
			s.setPhase("not_installed", "No model installed")
		}
		return
	}
	if stopped {
		s.setPhase("sleeping", "Stopped")
		return
	}
	if s.deferred.Load() { // started asleep and not yet woken: no model is loaded
		s.setPhase("sleeping", "Model not loaded")
		return
	}
	s.setPhase("starting", "Starting")
}

// Shutdown stops the agent first, then llama-server.
func (s *Supervisor) Shutdown() {
	s.Cancel()
	s.mu.Lock()
	lc := s.loadCancel
	s.mu.Unlock()
	if lc != nil {
		lc()
	}
	s.stopAgent()
	s.opMu.Lock()
	// said before the stop: a request that arrives now is answered at once, never proxied to a
	// server being stopped, and the held ones end (api answers "stopping" when the hub closes)
	s.setPhase("stopping", "Stopping")
	s.stopServerLocked()
	s.letGoContexts("the engine ends")
	s.opMu.Unlock()
	s.stop()
	s.card.stop()
	s.wg.Wait()
	s.cfg.Group.KillAll()
	s.hub.Close()
	if s.log != nil {
		s.log.Close()
	}
}

// Start starts the model server (local mode). It blocks until the server is
// healthy or has failed; the API calls it in the background.
func (s *Supervisor) Start() error {
	s.mu.Lock()
	s.userStopped = false
	local := s.st.Mode == "local"
	s.mu.Unlock()
	if s.deferred.Swap(false) { // the first start of an app that started asleep
		s.startAgent()
		if local {
			s.resumeUpgrade()
		}
	}
	if !local {
		return nil
	}
	if s.yielded() != nil {
		return ErrYielded
	}
	s.opMu.Lock()
	defer s.opMu.Unlock()
	if s.srv != nil && !s.srv.proc.Exited() {
		return nil
	}
	return s.startLocked()
}

// Stop stops the model server and keeps it stopped.
func (s *Supervisor) Stop() error {
	s.mu.Lock()
	s.userStopped = true
	lc := s.loadCancel
	s.mu.Unlock()
	if lc != nil {
		lc()
	}
	s.opMu.Lock()
	defer s.opMu.Unlock()
	local := s.Status().Mode == "local"
	_, installed := s.resolveInstall()
	if local && installed { // said before the stop: no request is proxied to a server being stopped
		s.setPhase("sleeping", "Stopped")
	}
	s.keepContexts("a stop", false)
	s.stopServerLocked()
	s.mu.Lock()
	s.yieldInfo = nil // a person's stop ends a yield; their start brings the watch back
	s.refreshLocked()
	s.mu.Unlock()
	if local && !installed && !s.customMissingPhase() {
		s.setPhase("not_installed", "No model installed")
	}
	return nil
}

// Sleep unloads the model; a proxied request or Wake loads it again.
func (s *Supervisor) Sleep() error {
	s.mu.Lock()
	lc := s.loadCancel
	local := s.st.Mode == "local"
	s.mu.Unlock()
	if !local || s.yielded() != nil {
		return nil
	}
	if lc != nil {
		lc()
	}
	s.opMu.Lock()
	defer s.opMu.Unlock()
	// said before the stop: a request that arrives during it is held, and wakes the model again
	s.setPhase("sleeping", "Model unloaded")
	s.keepContexts("a sleep", false)
	s.stopServerLocked()
	return nil
}

// Wake loads the model after Sleep.
func (s *Supervisor) Wake() error {
	defer func() { s.mu.Lock(); s.waking = false; s.mu.Unlock() }()
	return s.Start()
}

func (s *Supervisor) wake() { s.Wake() }

func (s *Supervisor) stopServerLocked() {
	srv := s.srv
	s.srv = nil
	s.ours.set(0)
	s.launching.set(0)
	if srv == nil {
		s.card.disarm() // a start that waited for the room to change waits no more
		return
	}
	s.mindDown(srv)
	srv.proc.Stop(s.cfg.StopGrace)
	s.mu.Lock()
	s.shape = nil
	s.llamaURL = ""
	s.refreshLocked()
	s.mu.Unlock()
}

// Settings returns the stored settings (the key is never returned).
func (s *Supervisor) Settings() Settings {
	s.mu.Lock()
	defer s.mu.Unlock()
	e := s.st.External
	e.APIKey = ""
	streamer := s.sf.streamer
	if streamer == "" {
		streamer = "on"
	}
	keepContext := s.sf.keepContext
	if keepContext == "" {
		keepContext = "on"
	}
	vision := s.vis.setting
	return Settings{Mode: s.st.Mode, External: e, LingerS: s.st.LingerS, KVType: s.st.KVType, SamplerPolicy: s.st.SamplerPolicy, Streamer: streamer, KeepContext: keepContext, Vision: &vision, VisionForced: s.cfg.Vision}
}

// LingerS is the stored linger setting.
func (s *Supervisor) LingerS() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.st.LingerS
}

// SetSettings saves settings (to settings.json); a mode change takes effect at once.
func (s *Supervisor) SetSettings(in Settings) error { return s.setSettings(in, true) }

// setSettings is SetSettings; persist says whether settings.json is written (not when the
// settings came from the file itself, which is then left exactly as the keeper wrote it).
func (s *Supervisor) setSettings(in Settings, persist bool) error {
	if persist { // the file's own edit (persist false) is kept as written; see ReloadSettings
		s.mu.Lock()
		turnsOff := s.vis.setting
		s.mu.Unlock()
		if turnsOff { // a change true -> false; an unchanged false is not a request
			if ce := s.CheckVisionOff(in.Vision); ce != nil {
				return ce
			}
		}
	}
	if in.Mode == "" {
		in.Mode = s.Settings().Mode
	}
	if in.Mode != "local" && in.Mode != "external" {
		return fmt.Errorf("mode must be local or external")
	}
	if in.KVType != "" {
		switch in.KVType {
		case "q4_0", "q8_0", "f16", "q4_1", "q5_0", "q5_1", "iq4_nl", "bf16", "f32":
		default:
			return fmt.Errorf("unknown kv_type %q", in.KVType)
		}
	}
	if in.SamplerPolicy != "" && !ValidPolicy(in.SamplerPolicy) {
		return fmt.Errorf("sampler_policy must be enforce, fill or off")
	}
	if in.LingerS < 0 {
		return fmt.Errorf("linger_s must not be negative")
	}
	if in.Streamer != "" && in.Streamer != "on" && in.Streamer != "off" {
		return fmt.Errorf("streamer must be on or off")
	}
	if in.KeepContext != "" && in.KeepContext != "on" && in.KeepContext != "off" && in.KeepContext != "always" {
		return fmt.Errorf("keep_context must be on, off or always")
	}
	s.mu.Lock()
	if in.KeepContext != "" {
		s.sf.keepContext = in.KeepContext
	}
	if in.Streamer != "" { // read by the proxy per request: applies to the next one
		s.sf.streamer = in.Streamer
	}
	old := s.st.Mode
	oldKV := s.st.KVType
	s.st.Mode = in.Mode
	s.st.External.BaseURL = strings.TrimSpace(in.External.BaseURL)
	s.st.External.Model = strings.TrimSpace(in.External.Model)
	if in.External.APIKey != "" {
		s.st.External.APIKey = in.External.APIKey
	}
	if in.ClearAPIKey {
		s.st.External.APIKey = ""
	}
	s.st.LingerS = in.LingerS
	if in.SamplerPolicy != "" && in.SamplerPolicy != s.st.SamplerPolicy {
		s.logf("sampler policy %s -> %s", s.st.SamplerPolicy, in.SamplerPolicy)
		s.st.SamplerPolicy = in.SamplerPolicy
	}
	if in.KVType != "" {
		s.st.KVType = in.KVType
	}
	kvChanged := s.st.KVType != oldKV
	visionChanged := in.Vision != nil && *in.Vision != s.vis.setting
	if visionChanged {
		s.vis.setting = *in.Vision
	}
	s.mu.Unlock()
	if persist {
		s.saveSettings()
	}
	switch {
	case old != in.Mode && in.Mode == "external":
		s.mu.Lock()
		lc := s.loadCancel
		s.mu.Unlock()
		if lc != nil {
			lc()
		}
		s.opMu.Lock()
		s.stopServerLocked()
		s.opMu.Unlock()
		s.setPhase("external", "Using your own endpoint")
		s.restartAgentInstalled()
	case old != in.Mode:
		s.restartAgent()
		go s.Start()
	case in.Mode == "external":
		s.restartAgentInstalled()
	case kvChanged:
		s.letGoContexts("the cache type changed")
		s.mu.Lock()
		s.reshape = true
		s.reshapeWhy = "KV type changed"
		s.mu.Unlock()
	}
	if visionChanged && in.Mode != "external" {
		s.visionWantChanged("the vision setting changed", true)
	}
	s.Nudge("settings") // after reshape is set: the reactor sees it when it wakes
	return nil
}

// ---- logs ----------------------------------------------------------------------------------

// Logs returns the last n lines of a log.
func (s *Supervisor) Logs(name string, n int) (string, error) {
	switch name {
	case "engine", "llama-server", "agent":
	default:
		return "", fmt.Errorf("unknown log %q", name)
	}
	if n <= 0 {
		n = 200
	}
	if n > 5000 {
		n = 5000
	}
	f, err := os.Open(filepath.Join(s.dir, "logs", name+".log"))
	if err != nil {
		return "", nil
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	ring := make([]string, 0, n)
	for sc.Scan() {
		if len(ring) == n {
			ring = ring[1:]
		}
		ring = append(ring, sc.Text())
	}
	return strings.Join(ring, "\n") + "\n", nil
}

// ---- small helpers -----------------------------------------------------------------------

func freePort() (int, error) {
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return 0, err
	}
	defer l.Close()
	return l.Addr().(*net.TCPAddr).Port, nil
}

func nvidiaMeasure(ctx context.Context, pid int) (int, error) {
	out, err := exec.CommandContext(ctx, "nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits").Output()
	if err != nil {
		return 0, err
	}
	total := 0
	for _, l := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		f := strings.Split(l, ",")
		if len(f) < 2 {
			continue
		}
		p, _ := strconv.Atoi(strings.TrimSpace(f[0]))
		m, _ := strconv.Atoi(strings.TrimSpace(f[1]))
		if p == pid {
			total += m
		}
	}
	if total == 0 {
		return 0, errors.New("pid not listed")
	}
	return total, nil
}

func bestGPU(hw probe.Hardware) (probe.GPU, bool) {
	var best probe.GPU
	ok := false
	for _, g := range hw.GPUs {
		if !ok || g.VRAMTotalMiB > best.VRAMTotalMiB {
			best, ok = g, true
		}
	}
	return best, ok
}

// FixExternalBase stores a corrected external base URL (after the Test found the /v1 fix).
func (s *Supervisor) FixExternalBase(b string) {
	s.mu.Lock()
	s.st.External.BaseURL = b
	s.mu.Unlock()
	s.saveSettings()
}
