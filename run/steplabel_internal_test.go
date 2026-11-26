package run

import (
	"testing"

	"hermesengine/fetch"
)

func TestStepLabelShowsTheVerifyPass(t *testing.T) {
	if got := stepLabel("model x", fetch.Progress{Step: "downloading"}); got != "model x" {
		t.Errorf("downloading: %q", got)
	}
	if got := stepLabel("model x", fetch.Progress{Step: "verifying"}); got != "verifying model x" {
		t.Errorf("verifying: %q", got)
	}
}
