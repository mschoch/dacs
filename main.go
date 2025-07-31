package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"path"
	"path/filepath"
	"strings"

	"github.com/ollama/ollama/api"
)

var (
	FALSE = false
	TRUE  = true
)

func main() {

	ctx := context.Background()

	var ollamaRawUrl string
	if ollamaRawUrl = os.Getenv("OLLAMA_HOST"); ollamaRawUrl == "" {
		ollamaRawUrl = "http://localhost:11434"
	}

	var toolsLLM string
	if toolsLLM = os.Getenv("TOOLS_LLM"); toolsLLM == "" {
		//toolsLLM = "llama3.1:8b"  // less vram
		//toolsLLM = "devstral:24b" // previous best
		toolsLLM = "qwen3:30b-a3b-instruct-2507-q4_K_M"
	}

	ollamaUrl, _ := url.Parse(ollamaRawUrl)
	client := api.NewClient(ollamaUrl, http.DefaultClient)

	scanner := bufio.NewScanner(os.Stdin)
	getUserMessage := func() (string, bool) {
		if !scanner.Scan() {
			return "", false
		}
		return scanner.Text(), true
	}

	tools := []Tool{
		ReadFileDefinition,
		ListFilesDefinition,
		EditFileDefinition,
	}
	agent := NewAgent(client, toolsLLM, getUserMessage, tools)
	err := agent.Run(ctx)
	if err != nil {
		fmt.Printf("Error: %s\n", err.Error())
	}
}

func NewAgent(
	client *api.Client,
	toolsLLM string,
	getUserMessage func() (string, bool),
	tools []Tool) *Agent {
	return &Agent{
		client:         client,
		toolsLLM:       toolsLLM,
		getUserMessage: getUserMessage,
		tools:          tools,
	}
}

type Agent struct {
	client         *api.Client
	toolsLLM       string
	getUserMessage func() (string, bool)
	tools          []Tool
}

func (a *Agent) Run(ctx context.Context) error {
	var conversation []api.Message

	conversation = append(conversation, api.Message{
		Role:    "system",
		Content: "You are an assistant with access to tools, if you do not have a tool to deal with the user's request but you think you can answer do it so, if not provide a list of the tools you do have.",
	})

	fmt.Printf("Chat with %s (use 'ctrl-c' to quit)\n", a.toolsLLM)

	readUserInput := true
	for {

		if readUserInput {
			fmt.Print("\u001b[94mYou\u001b[0m: ")
			userInput, ok := a.getUserMessage()
			if !ok {
				break
			}

			userMessage := api.Message{
				Role:    "user",
				Content: userInput,
			}
			conversation = append(conversation, userMessage)
		}

		res, err := a.runInference(ctx, conversation)
		if err != nil {
			return err
		}
		conversation = append(conversation, res.Message)

		if res.Message.Content != "" {
			fmt.Printf("\u001b[93mAgent\u001b[0m: %s\n", res.Message.Content)
		}

		var toolResults []api.Message
		for _, tc := range res.Message.ToolCalls {
			argsBuf, err2 := json.Marshal(tc.Function.Arguments)
			if err2 != nil {
				return fmt.Errorf("error marshaling json: %v", err2)
			}
			toolMsg, err3 := a.executeTool(tc.Function.Index, tc.Function.Name, argsBuf)
			if err3 != nil {
				return fmt.Errorf("error executing tool %s: %v", tc.Function.Name, err3)
			}

			toolUserMessage := api.Message{
				Role:    "user",
				Content: toolMsg,
			}
			toolResults = append(toolResults, toolUserMessage)
		}

		if len(toolResults) == 0 {
			readUserInput = true
			continue
		}
		readUserInput = false
		conversation = append(conversation, toolResults...)
	}

	return nil
}

func (a *Agent) executeTool(id int, name string, input json.RawMessage) (string, error) {
	var toolDef Tool
	var found bool
	for _, tool := range a.tools {
		if tool.Definition.Name == name {
			toolDef = tool
			found = true
			break
		}
	}
	if !found {
		return "", fmt.Errorf("tool %q not found", name)
	}

	fmt.Printf("\u001b[92mtool\u001b[0m: %s(%s)\n", name, input)
	response, err := toolDef.Function(input)
	if err != nil {
		return "", err
	}
	return response, nil
}

func (a *Agent) runInference(ctx context.Context, conversation []api.Message) (rv api.ChatResponse, err error) {
	var toolsList api.Tools
	for _, td := range a.tools {
		toolsList = append(toolsList, api.Tool{
			Type: "function",
			Function: api.ToolFunction{
				Name:        td.Definition.Name,
				Description: td.Definition.Description,
				Parameters:  td.Definition.Parameters,
			},
		})
	}

	err = a.client.Chat(ctx, &api.ChatRequest{
		Model:    a.toolsLLM,
		Messages: conversation,
		Options: map[string]interface{}{
			"temperature":   0.0,
			"repeat_last_n": 2,
		},
		Tools:  toolsList,
		Stream: &FALSE,
	}, func(resp api.ChatResponse) error {
		rv = resp
		return nil
	})

	return rv, err
}

type Tool struct {
	Definition api.ToolFunction
	Function   func(input json.RawMessage) (string, error)
}

var ReadFileDefinition = Tool{
	Definition: api.ToolFunction{
		Name:        "read_file",
		Description: "Read the contents of a given relative file path. Use this when you want to see what's inside a file. Do not use this with directory names.",
		Parameters: struct {
			Type       string   `json:"type"`
			Required   []string `json:"required"`
			Properties map[string]struct {
				Type        string   `json:"type"`
				Description string   `json:"description"`
				Enum        []string `json:"enum,omitempty"`
			} `json:"properties"`
		}(struct {
			Type       string
			Required   []string
			Properties map[string]struct {
				Type        string
				Description string
				Enum        []string
			}
		}{
			Type:     "object",
			Required: []string{},
			Properties: map[string]struct {
				Type        string
				Description string
				Enum        []string
			}{
				"path": {
					Type:        "string",
					Description: "The relative path of a file in the working directory.",
				},
			},
		}),
	},
	Function: ReadFile,
}

type ReadFileInput struct {
	Path string `json:"path"`
}

func ReadFile(input json.RawMessage) (string, error) {
	readFileInput := ReadFileInput{}
	err := json.Unmarshal(input, &readFileInput)
	if err != nil {
		panic(err)
	}

	content, err := os.ReadFile(readFileInput.Path)
	if err != nil {
		return "", err
	}
	return string(content), nil
}

// list

var ListFilesDefinition = Tool{
	Definition: api.ToolFunction{
		Name:        "list_files",
		Description: "List files and directories at a given path. If no path is provided, lists files in the current directory.",
		Parameters: struct {
			Type       string   `json:"type"`
			Required   []string `json:"required"`
			Properties map[string]struct {
				Type        string   `json:"type"`
				Description string   `json:"description"`
				Enum        []string `json:"enum,omitempty"`
			} `json:"properties"`
		}(struct {
			Type       string
			Required   []string
			Properties map[string]struct {
				Type        string
				Description string
				Enum        []string
			}
		}{
			Type:     "object",
			Required: []string{},
			Properties: map[string]struct {
				Type        string
				Description string
				Enum        []string
			}{
				"path": {
					Type:        "string",
					Description: "Optional relative path to list files from. Defaults to current directory if not provided.",
				},
			},
		}),
	},
	Function: ListFiles,
}

type ListFilesInput struct {
	Path string `json:"path,omitempty" jsonschema_description:"Optional relative path to list files from. Defaults to current directory if not provided."`
}

func ListFiles(input json.RawMessage) (string, error) {
	listFilesInput := ListFilesInput{}
	err := json.Unmarshal(input, &listFilesInput)
	if err != nil {
		panic(err)
	}

	dir := "."
	if listFilesInput.Path != "" {
		dir = listFilesInput.Path
	}

	var files []string
	err = filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		relPath, err := filepath.Rel(dir, path)
		if err != nil {
			return err
		}

		if relPath != "." {
			if info.IsDir() {
				files = append(files, relPath+"/")
			} else {
				files = append(files, relPath)
			}
		}
		return nil
	})

	if err != nil {
		return "", err
	}

	result, err := json.Marshal(files)
	if err != nil {
		return "", err
	}

	return string(result), nil
}

// edit

var EditFileDefinition = Tool{
	Definition: api.ToolFunction{
		Name: "edit_file",
		Description: `Make edits to a text file.

Replaces 'old_str' with 'new_str' in the given file. 'old_str' and 'new_str' MUST be different from each other.

If the file specified with path doesn't exist, it will be created.
`,
		Parameters: struct {
			Type       string   `json:"type"`
			Required   []string `json:"required"`
			Properties map[string]struct {
				Type        string   `json:"type"`
				Description string   `json:"description"`
				Enum        []string `json:"enum,omitempty"`
			} `json:"properties"`
		}(struct {
			Type       string
			Required   []string
			Properties map[string]struct {
				Type        string
				Description string
				Enum        []string
			}
		}{
			Type:     "object",
			Required: []string{},
			Properties: map[string]struct {
				Type        string
				Description string
				Enum        []string
			}{
				"path": {
					Type:        "string",
					Description: "The path to the file",
				},
				"old_str": {
					Type:        "string",
					Description: "Text to search for - must match exactly and must only have one match exactly",
				},
				"new_str": {
					Type:        "string",
					Description: "Text to replace old_str with",
				},
			},
		}),
	},
	Function: EditFile,
}

type EditFileInput struct {
	Path   string `json:"path"`
	OldStr string `json:"old_str"`
	NewStr string `json:"new_str"`
}

func EditFile(input json.RawMessage) (string, error) {
	editFileInput := EditFileInput{}
	err := json.Unmarshal(input, &editFileInput)
	if err != nil {
		return "", err
	}

	if editFileInput.Path == "" || editFileInput.OldStr == editFileInput.NewStr {
		return "", fmt.Errorf("invalid input parameters")
	}

	content, err := os.ReadFile(editFileInput.Path)
	if err != nil {
		if os.IsNotExist(err) && editFileInput.OldStr == "" {
			return createNewFile(editFileInput.Path, editFileInput.NewStr)
		}
		return "", err
	}

	oldContent := string(content)
	newContent := strings.Replace(oldContent, editFileInput.OldStr, editFileInput.NewStr, -1)

	if oldContent == newContent && editFileInput.OldStr != "" {
		return "", fmt.Errorf("old_str not found in file")
	}

	err = os.WriteFile(editFileInput.Path, []byte(newContent), 0644)
	if err != nil {
		return "", err
	}

	return "OK", nil
}

func createNewFile(filePath, content string) (string, error) {
	dir := path.Dir(filePath)
	if dir != "." {
		err := os.MkdirAll(dir, 0755)
		if err != nil {
			return "", fmt.Errorf("failed to create directory: %w", err)
		}
	}

	err := os.WriteFile(filePath, []byte(content), 0644)
	if err != nil {
		return "", fmt.Errorf("failed to create file: %w", err)
	}

	return fmt.Sprintf("Successfully created file %s", filePath), nil
}
