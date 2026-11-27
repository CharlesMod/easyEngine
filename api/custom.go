package api

import (
	"context"
	"net/http"

	"github.com/CharlesMod/easyEngine/run"
)

// customRoute serves custom/check, custom/use and custom/clear; it reports whether name was one.
func (a *API) customRoute(w http.ResponseWriter, r *http.Request, ctx context.Context, name string, need func(string) bool) bool {
	switch name {
	case "custom/check":
		if need("POST") {
			var in struct{ Path string }
			if err := decode(r, &in); err != nil {
				fail(w, 400, "body must be {\"path\": file}")
				return true
			}
			ck, err := a.S.CheckCustom(ctx, in.Path)
			if err != nil {
				fail(w, 500, err.Error())
				return true
			}
			writeJSON(w, 200, ck)
		}
	case "custom/use":
		if need("POST") {
			var in struct{ Path string }
			if err := decode(r, &in); err != nil || in.Path == "" {
				fail(w, 400, "body must be {\"path\": file}")
				return true
			}
			switch err := a.S.UseCustom(in.Path); err {
			case nil:
				writeJSON(w, 202, map[string]bool{"started": true})
			case run.ErrBusy:
				fail(w, 409, err.Error())
			default:
				fail(w, 400, err.Error())
			}
		}
	case "custom/clear":
		if need("POST") {
			a.S.ClearCustom()
			writeJSON(w, 200, map[string]string{"status": "cleared"})
		}
	default:
		return false
	}
	return true
}
