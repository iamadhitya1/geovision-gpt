import type { Metadata } from "next";
import { Bricolage_Grotesque, Geist, Geist_Mono } from "next/font/google";
import { AuthProvider } from "@/lib/auth";
import "./globals.css";

const display = Bricolage_Grotesque({
  variable: "--font-display",
  subsets: ["latin"],
  weight: ["500", "700"],
});
const body = Geist({ variable: "--font-body", subsets: ["latin"] });
const data = Geist_Mono({ variable: "--font-data", subsets: ["latin"] });

export const metadata: Metadata = {
  title: "GeoVisionGPT",
  description:
    "Ask questions about satellite imagery and compare two visual question answering models side by side.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className={`${display.variable} ${body.variable} ${data.variable} h-full antialiased`}>
      <body className="min-h-full flex flex-col">
        <AuthProvider>{children}</AuthProvider>
      </body>
    </html>
  );
}
