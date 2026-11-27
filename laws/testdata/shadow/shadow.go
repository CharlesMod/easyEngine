package shadow

type napper struct{}

func (napper) Sleep() {}

// No import of time here: a local named time is not the package.
func Local() {
	time := napper{}
	time.Sleep()
	other.Sleep()
}
