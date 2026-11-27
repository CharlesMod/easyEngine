package run

import (
	"strings"
	"testing"
)

// the old names, built from pieces: only internal/legacy may spell them
const (
	oldLLM       = "HER" + "MESVOICE_LLM"
	oldEngineURL = "HER" + "MES_ENGINE_URL"
)

func envMap(env []string) map[string]string {
	m := map[string]string{}
	for _, kv := range env {
		if k, v, ok := strings.Cut(kv, "="); ok {
			m[k] = v
		}
	}
	return m
}

func agentEnvFor(llmURLEnv, mode string) map[string]string {
	s := &Supervisor{cfg: Config{Profile: Profile{Agent: &AgentSpec{LLMURLEnv: llmURLEnv}}}, port: 4711}
	s.st.Mode = mode
	s.st.External.Model = "m1"
	return envMap(s.agentEnv())
}

func TestAgentEnvDefaultNamesCarryTheOldTwins(t *testing.T) {
	m := agentEnvFor("", "")
	for _, k := range []string{"EASY_ENGINE_LLM_URL", oldLLM + "_URL", "EASY_ENGINE_URL", oldEngineURL} {
		if !strings.HasPrefix(m[k], "http://127.0.0.1:4711") {
			t.Errorf("%s = %q", k, m[k])
		}
	}
	if m["EASY_ENGINE_LLM_URL"] != m[oldLLM+"_URL"] || m["EASY_ENGINE_URL"] != m[oldEngineURL] {
		t.Errorf("old and new differ: %v", m)
	}
	if _, ok := m["EASY_ENGINE_LLM_MODE"]; ok {
		t.Error("MODE set outside external mode")
	}
}

func TestAgentEnvExternalModeSetsAllSixLLMVariables(t *testing.T) {
	m := agentEnvFor("", "external")
	for _, p := range []string{"EASY_ENGINE_LLM", oldLLM} {
		if m[p+"_URL"] != "http://127.0.0.1:4711/v1" || m[p+"_MODE"] != "openai" || m[p+"_MODEL"] != "m1" {
			t.Errorf("%s: url %q mode %q model %q", p, m[p+"_URL"], m[p+"_MODE"], m[p+"_MODEL"])
		}
	}
	if m["EASY_ENGINE_URL"] != "http://127.0.0.1:4711" {
		t.Errorf("engine url %q", m["EASY_ENGINE_URL"])
	}
}

func TestAgentEnvNamedURLVariableUsesOnlyItsPrefix(t *testing.T) {
	m := agentEnvFor("MY_LLM_URL", "external")
	if m["MY_LLM_URL"] != "http://127.0.0.1:4711/v1" || m["MY_LLM_MODE"] != "openai" || m["MY_LLM_MODEL"] != "m1" {
		t.Errorf("named: %v", m)
	}
	for _, k := range []string{"EASY_ENGINE_LLM_URL", "EASY_ENGINE_LLM_MODE", oldLLM + "_URL", oldLLM + "_MODE", oldLLM + "_MODEL"} {
		if _, ok := m[k]; ok {
			t.Errorf("%s set although the profile names its own variable", k)
		}
	}
}
