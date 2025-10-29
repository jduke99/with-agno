"use client";

import { useCopilotAction } from "@copilotkit/react-core";
import { CopilotKitCSSProperties, CopilotChat } from "@copilotkit/react-ui";
import { useState } from "react";
import type { ActionRenderProps } from "@copilotkit/react-core";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
  PointElement,
  LineElement,
} from 'chart.js';
import { Bar, Doughnut, Line, Pie } from 'react-chartjs-2';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
  PointElement,
  LineElement
);

export default function CopilotKitPage() {
  const [themeColor, setThemeColor] = useState("#1e40af");

  // Frontend action for displaying charts inline in chat
  useCopilotAction({
    name: "display_chart",
    description: "Display interactive charts inline in the chat",
    parameters: [{
      name: "chart_type",
      description: "Type of chart (bar, line, pie, doughnut)",
      type: "string",
      required: true,
      enum: ["bar", "line", "pie", "doughnut"],
    }, {
      name: "title",
      description: "Chart title",
      type: "string",
      required: true,
    }, {
      name: "labels",
      description: "Chart labels as comma-separated string",
      type: "string",
      required: true,
    }, {
      name: "data",
      description: "Chart data values as comma-separated numbers",
      type: "string",
      required: true,
    }, {
      name: "dataset_label",
      description: "Label for the data series",
      type: "string",
      required: false,
    }],
    render: ({ status, args }) => {
      if (status === "executing") {
        return (
          <div className="my-4 p-4 bg-blue-50 border border-blue-200 rounded-lg">
            <div className="flex items-center gap-2">
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
              <span className="text-blue-700">Generating chart...</span>
            </div>
          </div>
        );
      }
      if (status === "complete" && args) {
        // Parse comma-separated strings into arrays
        const labels = typeof args.labels === 'string' ? args.labels.split(',').map(s => s.trim()) : [];
        const data = typeof args.data === 'string' ? args.data.split(',').map(s => parseFloat(s.trim())) : [];

        const chartData = {
          labels: labels,
          datasets: [{
            label: args.dataset_label || 'Data',
            data: data,
            backgroundColor: [
              'rgba(59, 130, 246, 0.8)',
              'rgba(16, 185, 129, 0.8)',
              'rgba(245, 158, 11, 0.8)',
              'rgba(239, 68, 68, 0.8)',
              'rgba(139, 92, 246, 0.8)',
              'rgba(236, 72, 153, 0.8)',
            ],
            borderColor: [
              'rgba(59, 130, 246, 1)',
              'rgba(16, 185, 129, 1)',
              'rgba(245, 158, 11, 1)',
              'rgba(239, 68, 68, 1)',
              'rgba(139, 92, 246, 1)',
              'rgba(236, 72, 153, 1)',
            ],
            borderWidth: 2,
          }],
        };

        const options = {
          responsive: true,
          plugins: {
            legend: {
              position: 'top' as const,
            },
            title: {
              display: true,
              text: args.title || 'Chart',
              font: {
                size: 16,
                weight: 'bold' as const,
              },
            },
          },
          scales: args.chart_type === 'bar' || args.chart_type === 'line' ? {
            y: {
              beginAtZero: true,
            },
          } : undefined,
        };

        return (
          <div className="my-4 p-4 bg-white border border-gray-200 rounded-lg shadow-sm">
            <div className="w-full max-w-2xl mx-auto">
              {args.chart_type === 'bar' && <Bar data={chartData} options={options} />}
              {args.chart_type === 'line' && <Line data={chartData} options={options} />}
              {args.chart_type === 'pie' && <Pie data={chartData} options={options} />}
              {args.chart_type === 'doughnut' && <Doughnut data={chartData} options={options} />}
            </div>
          </div>
        );
      }
      return <></>;
    },
  });

  return (
    <main style={{ "--copilot-kit-primary-color": themeColor } as CopilotKitCSSProperties} className="h-screen flex flex-col">
      {/* Header */}
      <header className="bg-blue-900 text-white p-6 shadow-lg">
        <h1 className="text-3xl font-bold text-center">
          Georgia APCD Provider Workforce Explorer
        </h1>
      </header>

      {/* Main Content Area */}
      <div className="flex-1 flex justify-center p-4">
        {/* CopilotChat - Main Interaction */}
        <div className="w-full max-w-6xl h-full">
          <CopilotChat
            className="h-full"
            labels={{
              initial: "👋 Welcome to the Georgia APCD Provider Workforce Explorer!\n\nI'm here to help you analyze provider workforce data. You can ask me questions about:\n\n- Provider demographics and distribution\n- Workforce trends and patterns  \n- Data visualizations and charts\n\nTry asking: \"Show me a test chart\" to see the visualization capabilities!"
            }}
          />
        </div>
      </div>
    </main>
  );
}
