package main

import (
	"net/http"

	"github.com/alecthomas/kong"
)

type InstallCmd struct {
	URL string `arg:"" name:"url" help:"The URL of the compressed Paddle Inference C Library."`

	Out     string `short:"o" help:"The output directory."`
	Force   bool   `short:"f" help:"Force to download Paddle Inference C Library (even if it already exists)."`
	Verbose bool   `short:"v" help:"Print the progress messages."`
}

func (cmd *InstallCmd) Run() error {
	i := Installer{
		HTTPClient: http.DefaultClient,
		Verbose:    cmd.Verbose,
	}
	return i.Install(cmd.URL, cmd.Out, cmd.Force)
}

var CLI struct {
	Install InstallCmd `cmd:"" help:"Install Paddle Inference Go API."`
}

func main() {
	ctx := kong.Parse(&CLI)
	err := ctx.Run()
	ctx.FatalIfErrorf(err)
}
