package run

// SetAgentNoVenv switches the agent layout (no venv: base Python + <env>/site) for a test; it returns the undo.
func SetAgentNoVenv(on bool) func() {
	old := agentNoVenv
	agentNoVenv = on
	return func() { agentNoVenv = old }
}
