import type { Metadata } from "next";

import { CopilotKit } from "@copilotkit/react-core";
import "./globals.css";
import "@copilotkit/react-ui/styles.css";

export const metadata: Metadata = {
  title: "Georgia APCD Provider Workforce Explorer",
  description: "AI-powered analysis and visualization of Georgia provider workforce data",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={"antialiased"}>
        <CopilotKit runtimeUrl="/api/copilotkit" agent="workforce_agent">
          {children}
        </CopilotKit>
      </body>
    </html>
  );
}
