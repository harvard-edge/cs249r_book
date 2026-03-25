import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Hardware Simulator",
};

export default function Layout({ children }: { children: React.ReactNode }) {
  return children;
}
