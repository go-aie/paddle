package main

import (
	"archive/tar"
	"bufio"
	"compress/gzip"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/schollz/progressbar/v3"
)

const (
	paddleModPath = "github.com/paddlepaddle/paddle/paddle/fluid/inference/goapi"
)

type Installer struct {
	HTTPClient *http.Client
	Verbose    bool
}

func (i Installer) Install(srcURL, dstDir string, force bool) error {
	if !exists(dstDir) {
		if err := os.Mkdir(dstDir, os.ModePerm); err != nil {
			return err
		}
	}

	paddleInferenceGzip := filepath.Join(dstDir, parseFilenameFromURL(srcURL))
	paddleInferenceDir := strings.TrimSuffix(paddleInferenceGzip, filepath.Ext(paddleInferenceGzip))

	if force || !exists(paddleInferenceDir) {
		if err := i.download(paddleInferenceGzip, srcURL); err != nil {
			return err
		}
		if err := i.decompress(dstDir, paddleInferenceGzip); err != nil {
			return err
		}
	}

	return i.updatePaddleInference(paddleInferenceDir)
}

func (i Installer) download(dstPath, srcURL string) error {
	req, err := http.NewRequest(http.MethodGet, srcURL, nil)
	if err != nil {
		return err
	}

	resp, err := i.HTTPClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	f, err := os.OpenFile(dstPath, os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return err
	}
	defer f.Close()

	var writer io.Writer = f
	if i.Verbose {
		desc := "Downloading " + filepath.Base(dstPath)
		bar := newBytesBar(resp.ContentLength, desc)
		writer = io.MultiWriter(f, bar)
	}

	_, err = io.Copy(writer, resp.Body)
	return err
}

// Decompress decompresses the gzip file src into a directory dst.
//
// This code is borrowed from https://github.com/mimoo/eureka/blob/99112743244fca318cabe54bfb91cf8bf6d7dc33/folders.go#L96.
func (i Installer) decompress(dst, src string) error {
	if i.Verbose {
		fmt.Printf("Decompressing %s into %s\n", src, dst)
	}

	// Open the gzipped file
	gzipFile, err := os.Open(src)
	if err != nil {
		return err
	}
	defer gzipFile.Close()

	// gunzip
	gr, err := gzip.NewReader(gzipFile)
	if err != nil {
		return err
	}
	defer gr.Close()

	// untar
	tr := tar.NewReader(gr)

	// uncompress each element
	for {
		header, err := tr.Next()
		if err == io.EOF {
			break // End of archive
		}
		if err != nil {
			return err
		}
		target := header.Name

		// validate name against path traversal
		if !validRelPath(header.Name) {
			return fmt.Errorf("tar contained invalid name error %q\n", target)
		}

		// add dst + re-format slashes according to system
		target = filepath.Join(dst, header.Name)
		// if no join is needed, replace with ToSlash:
		// target = filepath.ToSlash(header.Name)

		// check the type
		switch header.Typeflag {

		// if its a dir and it doesn't exist create it (with 0755 permission)
		case tar.TypeDir:
			if _, err := os.Stat(target); err != nil {
				if err := os.MkdirAll(target, 0755); err != nil {
					return err
				}
			}
		// if it's a file create it (with same permission)
		case tar.TypeReg:
			fileToWrite, err := os.OpenFile(target, os.O_CREATE|os.O_RDWR, os.FileMode(header.Mode))
			if err != nil {
				return err
			}
			// copy over contents
			if _, err := io.Copy(fileToWrite, tr); err != nil {
				return err
			}
			// manually close here after each file operation; defering would cause each file close
			// to wait until all operations have completed.
			fileToWrite.Close()
		}
	}

	return nil
}

func (i Installer) updatePaddleInference(paddleInferenceDir string) error {
	// Get the commit ID.
	metadata, err := parseVersionMetadata(paddleInferenceDir)
	if err != nil {
		return err
	}
	commitID := metadata["GIT COMMIT ID"]

	// Update the version of paddle inference Go API.
	if i.Verbose {
		fmt.Println("Installing", paddleModPath+"@"+commitID)
	}
	mustRunCmd("go", "get", "-u", paddleModPath+"@"+commitID)

	// Get the final version number.
	//
	// Example output:
	// 	github.com/paddlepaddle/paddle/paddle/fluid/inference/goapi v0.0.0-20221202080524-4743cc8b9a8d
	output := mustRunCmd("go", "list", "-m", paddleModPath)
	subs := strings.Fields(output)
	version := subs[len(subs)-1]

	// Create a new symbolic link which points to the paddle inference C library.
	target, err := filepath.Abs(paddleInferenceDir)
	if err != nil {
		return err
	}
	output = mustRunCmd("go", "env", "GOMODCACHE")
	modCachePath := strings.TrimSpace(output)
	symlink := filepath.Join(modCachePath, paddleModPath+"@"+version, "paddle_inference_c")
	if exists(symlink) {
		_ = os.Remove(symlink)
	}

	if i.Verbose {
		fmt.Printf("Creating the symbolic link: %s -> %s\n", symlink, target)
	}
	return os.Symlink(target, symlink)
}

func newBytesBar(max int64, desc string) *progressbar.ProgressBar {
	return progressbar.NewOptions64(
		max,
		progressbar.OptionSetDescription(desc),
		progressbar.OptionSetWriter(os.Stderr),
		progressbar.OptionEnableColorCodes(true),
		progressbar.OptionShowBytes(true),
		//progressbar.OptionSetWidth(80),
		progressbar.OptionThrottle(65*time.Millisecond),
		progressbar.OptionShowCount(),
		progressbar.OptionOnCompletion(func() { fmt.Fprint(os.Stderr, "\n") }),
		progressbar.OptionSpinnerType(14),
		//progressbar.OptionFullWidth(),
		progressbar.OptionSetRenderBlankState(true),
	)
}

func parseFilenameFromURL(s string) string {
	u, err := url.Parse(s)
	if err != nil {
		panic(err)
	}
	return filepath.Base(u.Path)
}

func parseVersionMetadata(paddleInferenceDir string) (map[string]string, error) {
	filename := filepath.Join(paddleInferenceDir, "version.txt")
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	metadata := make(map[string]string)

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		subs := strings.SplitN(scanner.Text(), ":", 2)
		if len(subs) != 2 {
			continue
		}

		k, v := strings.TrimSpace(subs[0]), strings.TrimSpace(subs[1])
		metadata[k] = v
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	return metadata, nil
}

func validRelPath(p string) bool {
	if p == "" || strings.Contains(p, `\`) || strings.HasPrefix(p, "/") || strings.Contains(p, "../") {
		return false
	}
	return true
}

func mustRunCmd(name string, args ...string) string {
	cmd := exec.Command(name, args...)
	output, err := cmd.CombinedOutput()
	if err != nil {
		panic(fmt.Errorf("err: %s", output))
	}
	return string(output)
}

func exists(name string) bool {
	_, err := os.Lstat(name)
	return err == nil
}
