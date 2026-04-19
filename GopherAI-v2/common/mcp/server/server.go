package mcp

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/mark3labs/mcp-go/server"
)

const amapAPIKey = "b050c5d7476435dc6c4b8e8bdb2d2b2f"

type amapResponse struct {
	Status string `json:"status"`
	Info   string `json:"info"`
	Lives  []struct {
		Province     string `json:"province"`
		City         string `json:"city"`
		Weather      string `json:"weather"`
		Temperature  string `json:"temperature"`
		WindDirection string `json:"winddirection"`
		WindPower    string `json:"windpower"`
		Humidity     string `json:"humidity"`
		ReportTime   string `json:"reporttime"`
	} `json:"lives"`
}

type WeatherAPIClient struct{}

func NewWeatherAPIClient() *WeatherAPIClient {
	return &WeatherAPIClient{}
}

func (c *WeatherAPIClient) GetWeather(ctx context.Context, city string) (string, error) {
	apiURL := fmt.Sprintf(
		"https://restapi.amap.com/v3/weather/weatherInfo?city=%s&key=%s&extensions=base",
		url.QueryEscape(city), amapAPIKey,
	)

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, apiURL, nil)
	if err != nil {
		return "", fmt.Errorf("create request failed: %w", err)
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("http request failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("read response failed: %w", err)
	}

	var result amapResponse
	if err := json.Unmarshal(body, &result); err != nil {
		return "", fmt.Errorf("json parse failed: %w", err)
	}
	if result.Status != "1" || len(result.Lives) == 0 {
		return "", fmt.Errorf("city not found or api error: %s", result.Info)
	}

	live := result.Lives[0]
	return fmt.Sprintf(
		"城市: %s %s\n天气: %s\n温度: %s°C\n湿度: %s%%\n风向: %s\n风力: %s级\n更新时间: %s",
		live.Province, live.City,
		live.Weather,
		live.Temperature,
		live.Humidity,
		live.WindDirection,
		live.WindPower,
		live.ReportTime,
	), nil
}

/*
========================
MCP Server
========================
*/

func NewMCPServer() *server.MCPServer {
	weatherClient := NewWeatherAPIClient()

	mcpServer := server.NewMCPServer(
		"weather-query-server",
		"1.0.0",
		server.WithToolCapabilities(true),
		server.WithLogging(),
	)

	mcpServer.AddTool(
		mcp.NewTool(
			"get_weather",
			mcp.WithDescription("获取指定城市的天气信息"),
			mcp.WithString(
				"city",
				mcp.Description("城市名称，支持中文，如 北京、上海、广州"),
				mcp.Required(),
			),
		),
		func(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			args := request.GetArguments()
			city, ok := args["city"].(string)
			if !ok || city == "" {
				return nil, fmt.Errorf("invalid city argument")
			}

			result, err := weatherClient.GetWeather(ctx, city)
			if err != nil {
				return nil, err
			}

			return &mcp.CallToolResult{
				Content: []mcp.Content{
					mcp.TextContent{
						Type: "text",
						Text: result,
					},
				},
			}, nil
		},
	)

	return mcpServer
}

// StartServer 启动MCP服务器
func StartServer(httpAddr string) error {
	mcpServer := NewMCPServer()
	httpServer := server.NewStreamableHTTPServer(mcpServer)
	log.Printf("HTTP MCP server listening on %s/mcp", httpAddr)
	return httpServer.Start(httpAddr)
}
