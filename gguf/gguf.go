// STUB: replaced by eng/fetch
package gguf

import "errors"

type Info struct {
	Arch                                                                                string
	NLayer, NKVHead, HeadDimK, HeadDimV, SlidingWindow, SWALayers, NExpert, NExpertUsed int
	ExpertBytes, DenseBytes, FileSize                                                   int64
}

func Read(path string) (Info, error) { return Info{}, errors.New("gguf stub") }
