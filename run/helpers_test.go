package run_test

import (
	"net/http"
	"time"

	"github.com/CharlesMod/easyEngine/run"
)

func withShortHTTP(c run.Config) run.Config {
	c.HTTP = &http.Client{Timeout: 150 * time.Millisecond}
	return c
}
