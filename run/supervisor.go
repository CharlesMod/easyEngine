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

	"hermesengine/catalog"
	"hermesengine/fit"
	"hermesengine/lifecycle"
	"hermesengine/probe"
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
}

// ExternalInfo is the external endpoint without its key.
type ExternalInfo struct {
	BaseURL   string `json:"base_url"`
	Model     string `json:"model"`
	APIKeySet bool   `json:"api_key_set"`
}

// Status is GET /engine/v1/status.
type Status struct {
	Version   string       `json:"version"`
	Port      int          `json:"port"`
	Mode      string       `json:"mode"`
	Phase     string       `json:"phase"`
	Message   string       `json:"message"`
	Progress  *Progress    `json:"progress"`
	Model     string       `json:"model"`
	Backend   string       `json:"backend"`
	Shape     *ShapeInfo   `json:"shape"`
	ServerURL string       `json:"server_url"`
	Agent     AgentStatus  `json:"agent"`
	Parents   []Parent     `json:"parents"`
	TokS      *float64     `json:"tok_s"`
	Errors    []string     `json:"errors"`
	External  ExternalInfo `json:"external"`
	LingerS   int          `json:"linger_s"`
	KVType    string       `json:"kv_type"`
	Restarts  int          `json:"restarts"`
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
}

// Route tells the proxy where a request goes.
type Route struct {
	Kind    string // local | external | unavailable
	URL     string // base URL (no trailing slash); for external it includes any /v1
	Key     string
	Message string
}

type server struct {
	proc      *lifecycle.Proc
	port      int
	shape     fit.Shape
	key       string
	fp        fit.Footprint
	startedAt time.Time
	misses    int
	idleCount int
	idleSince time.Time
	nextGrow  time.Time
	measured  int
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
	retryAt     time.Time
	retryable   bool
	reshape     bool
	tokS        float64

	inflight int32

	opMu sync.Mutex
	srv  *server

	agent agentMgr
	wg    sync.WaitGroup
}

// New builds a supervisor. Call Run to start the watchdog.
func New(cfg Config) (*Supervisor, error) {
	cfg.defaults()
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
	s := &Supervisor{cfg: cfg, dir: cfg.DataDir, root: ctx, stop: cancel, log: lg, st: loadState(cfg.DataDir)}
	s.agent.s = s
	s.setPhaseLocked("not_installed", "No model installed")
	return s, nil
}

// SetPort records the control API port (the agent is pointed at it).
func (s *Supervisor) SetPort(p int) {
	s.mu.Lock()
	s.port = p
	s.mu.Unlock()
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
}

func (s *Supervisor) setPhaseLocked(phase, msg string) {
	s.phase, s.message = phase, msg
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
	s.mu.Lock()
	defer s.mu.Unlock()
	st := Status{
		Version: s.cfg.Version, Port: s.port, Mode: s.st.Mode, Phase: s.phase, Message: s.message,
		Model: s.st.Model, Backend: s.st.Backend, Errors: append([]string{}, s.errs...),
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
	return st
}

func (s *Supervisor) save() {
	s.mu.Lock()
	st := s.st
	st.Book.Entries = copyBook(s.st.Book.Entries)
	s.mu.Unlock()
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

func (s *Supervisor) wants() fit.Wants {
	p := s.cfg.Profile
	s.mu.Lock()
	kv := s.st.KVType
	s.mu.Unlock()
	return fit.Wants{Slots: p.Slots, MinSlots: p.MinSlots, CtxMax: p.CtxPerSlot.Max, CtxFloor: p.CtxPerSlot.Floor, KVType: kv}
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
func (s *Supervisor) Plan(ctx context.Context) (fit.PlanResult, error) {
	hw, err := s.Hardware(ctx, false)
	if err != nil {
		return fit.PlanResult{}, err
	}
	cat, err := s.catalog()
	if err != nil {
		return fit.PlanResult{}, err
	}
	return fit.Plan(cat, hw, s.wants(), s.cfg.Profile.GameReserveMiB, s.modelInstalled), nil
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
}

func (s *Supervisor) resolveInstall() (install, bool) {
	s.mu.Lock()
	id, be := s.st.Model, s.st.Backend
	s.mu.Unlock()
	if id == "" || be == "" {
		return install{}, false
	}
	cat, err := s.catalog()
	if err != nil {
		return install{}, false
	}
	m, ok := cat.Model(id)
	if !ok || !s.modelInstalled(m) {
		return install{}, false
	}
	bin := findServer(filepath.Join(s.dir, "backends", be))
	if bin == "" {
		return install{}, false
	}
	return install{model: m, modelPath: s.modelPath(m), bin: bin, backend: be}, true
}

// ---- proxy helpers ------------------------------------------------------------------------

// Begin marks a proxied request in flight; call the result when it ends.
func (s *Supervisor) Begin() func() {
	atomic.AddInt32(&s.inflight, 1)
	var once sync.Once
	return func() { once.Do(func() { atomic.AddInt32(&s.inflight, -1) }) }
}

// Route says where the proxy should send a request.
func (s *Supervisor) Route() Route {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.st.Mode == "external" {
		return Route{Kind: "external", URL: strings.TrimRight(s.st.External.BaseURL, "/"), Key: s.st.External.APIKey}
	}
	switch s.phase {
	case "ready":
		return Route{Kind: "local", URL: s.llamaURL}
	case "sleeping":
		if !s.userStopped && !s.waking {
			s.waking = true
			go s.wake()
		}
		return Route{Kind: "unavailable", Message: "Loading model"}
	case "not_installed":
		return Route{Kind: "unavailable", Message: "No model installed"}
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
	s.mu.Unlock()
}

// ---- lifecycle -----------------------------------------------------------------------------

// Run starts the watchdog and, in local mode with an installed model, the server.
// It returns at once; Shutdown ends everything.
func (s *Supervisor) Run() {
	s.settle()
	if !s.cfg.NoAutoStart {
		s.startAgent()
		s.mu.Lock()
		local := s.st.Mode == "local"
		s.mu.Unlock()
		if local {
			s.wg.Add(1)
			go func() { defer s.wg.Done(); s.Start() }()
		}
	}
	s.wg.Add(1)
	go func() { defer s.wg.Done(); s.watchdog() }()
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
		s.setPhase("not_installed", "No model installed")
		return
	}
	if stopped {
		s.setPhase("sleeping", "Stopped")
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
	s.stopServerLocked()
	s.opMu.Unlock()
	s.stop()
	s.wg.Wait()
	s.cfg.Group.KillAll()
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
	if !local {
		return nil
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
	s.stopServerLocked()
	if s.Status().Mode == "local" {
		if _, ok := s.resolveInstall(); ok {
			s.setPhase("sleeping", "Stopped")
		} else {
			s.setPhase("not_installed", "No model installed")
		}
	}
	return nil
}

// Sleep unloads the model; a proxied request or Wake loads it again.
func (s *Supervisor) Sleep() error {
	s.mu.Lock()
	lc := s.loadCancel
	local := s.st.Mode == "local"
	s.mu.Unlock()
	if !local {
		return nil
	}
	if lc != nil {
		lc()
	}
	s.opMu.Lock()
	defer s.opMu.Unlock()
	s.stopServerLocked()
	s.setPhase("sleeping", "Model unloaded")
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
	if srv == nil {
		return
	}
	srv.proc.Stop(s.cfg.StopGrace)
	s.mu.Lock()
	s.shape = nil
	s.llamaURL = ""
	s.mu.Unlock()
}

// Settings returns the stored settings (the key is never returned).
func (s *Supervisor) Settings() Settings {
	s.mu.Lock()
	defer s.mu.Unlock()
	e := s.st.External
	e.APIKey = ""
	return Settings{Mode: s.st.Mode, External: e, LingerS: s.st.LingerS, KVType: s.st.KVType}
}

// LingerS is the stored linger setting.
func (s *Supervisor) LingerS() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.st.LingerS
}

// SetSettings saves settings; a mode change takes effect at once.
func (s *Supervisor) SetSettings(in Settings) error {
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
	if in.LingerS < 0 {
		return fmt.Errorf("linger_s must not be negative")
	}
	s.mu.Lock()
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
	if in.KVType != "" {
		s.st.KVType = in.KVType
	}
	kvChanged := s.st.KVType != oldKV
	s.mu.Unlock()
	s.save()
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
		s.restartAgent()
	case old != in.Mode:
		s.restartAgent()
		go s.Start()
	case in.Mode == "external":
		s.restartAgent()
	case kvChanged:
		s.mu.Lock()
		s.reshape = true
		s.mu.Unlock()
	}
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
	s.save()
}
