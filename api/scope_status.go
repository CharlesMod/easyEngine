package api

import (
	"net/http"

	"github.com/CharlesMod/easyEngine/internal/tokens"
	"github.com/CharlesMod/easyEngine/run"
)

// PairedStatus is GET /engine/v1/status as a paired token sees it: the same JSON names and
// shapes as run.Status, with fields removed, never renamed. It is built by copying the fields
// listed here into a fresh value, so a field added to run.Status later is withheld from a
// paired token until it is added here on purpose (a test fails until it is classified).
type PairedStatus struct {
	Version        string          `json:"version"`
	API            int             `json:"api"`
	Mode           string          `json:"mode"`
	Phase          string          `json:"phase"`
	Message        string          `json:"message"`
	Progress       *run.Progress   `json:"progress"`
	Model          string          `json:"model"`
	Backend        string          `json:"backend"`
	Shape          *run.ShapeInfo  `json:"shape"`
	TokS           *float64        `json:"tok_s"`
	ModelTier      string          `json:"model_tier"`
	ModelStage     string          `json:"model_stage"`
	KVType         string          `json:"kv_type"`
	Restarts       int             `json:"restarts"`
	ModelFallback  bool            `json:"model_fallback"`
	ModelNote      string          `json:"model_note,omitempty"`
	Sampler        run.SamplerInfo `json:"sampler"`
	ModelName      string          `json:"model_name"`
	ModelCodename  string          `json:"model_codename"`
	ModelParams    int64           `json:"model_params"`
	ModelSizeClass string          `json:"model_size_class"`
	Held           int             `json:"held"`
}

// StatusKept lists the run.Status JSON names a paired token sees; StatusRemoved the ones it does
// not. Every run.Status field is in exactly one (TestStatusFieldsAreClassified).
var (
	StatusKept = []string{"version", "api", "mode", "phase", "message", "progress", "model", "backend",
		"shape", "tok_s", "model_tier", "model_stage", "kv_type", "restarts", "model_fallback",
		"model_note", "sampler", "model_name", "model_codename", "model_params", "model_size_class", "held"}
	StatusRemoved = []string{
		"port",       // the control API's own port
		"server_url", // the local model server's URL and port
		"agent",      // the agent's pid and detail
		"parents",    // watched parent pids
		"errors",     // error texts can name paths and user names
		"external",   // the owner's external endpoint
		"linger_s",   // owner setting
		"upgrade",    // carries error text
		"contexts",   // saved slots, with other clients' ids
		"custom",     // the owner's own file path
		"previous",   // the previous model's file
		"gpu",        // adapter ids and names
		"device",     // llama-server device pin: env and args
		"recovery",   // last error text
		"yield",      // names the program the card yielded to
		"build",      // the llama.cpp build and its move: the move's error text can name paths
	}
)

// PairedStatusOf copies the kept fields. message is blanked in the phases whose text carries a
// path, an error or the name of another program ("error", "yielded"); the phase says the rest.
func PairedStatusOf(st run.Status) PairedStatus {
	msg := st.Message
	if st.Phase == "error" || st.Phase == "yielded" {
		msg = ""
	}
	return PairedStatus{
		Version: st.Version, API: st.API, Mode: st.Mode, Phase: st.Phase, Message: msg,
		Progress: st.Progress, Model: st.Model, Backend: st.Backend, Shape: st.Shape, TokS: st.TokS,
		ModelTier: st.ModelTier, ModelStage: st.ModelStage, KVType: st.KVType, Restarts: st.Restarts,
		ModelFallback: st.ModelFallback, ModelNote: st.ModelNote, Sampler: st.Sampler,
		ModelName: st.ModelName, ModelCodename: st.ModelCodename, ModelParams: st.ModelParams, ModelSizeClass: st.ModelSizeClass,
		Held: st.Held,
	}
}

// statusFor is what the status endpoint writes: the full status, or the reduced one for a
// paired token (a client id that is none of the master, loopback, page or open callers).
func statusFor(r *http.Request, st run.Status) any {
	switch id := ClientID(r.Context()); id {
	case "", tokens.MasterID, ClientLoopback, ClientPage, ClientOpen:
		return st
	}
	return PairedStatusOf(st)
}
