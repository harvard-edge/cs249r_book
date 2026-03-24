"use client";

import { useState, useMemo } from "react";
import { Search, ChevronUp, ChevronDown, AlertCircle } from "lucide-react";
import clsx from "clsx";
import {
  type Concept,
  search as searchConcepts,
  getChapters,
  getTracks,
  getRoles,
  formatChapter,
} from "@/lib/taxonomy";

type SortKey = "name" | "tracks" | "question_count" | "role" | "chapter";
type SortDir = "asc" | "desc";

interface Props {
  onSelect: (concept: Concept) => void;
  selected: string | null;
}

export default function ConceptBrowser({ onSelect, selected }: Props) {
  const [query, setQuery] = useState("");
  const [sortKey, setSortKey] = useState<SortKey>("name");
  const [sortDir, setSortDir] = useState<SortDir>("asc");
  const [filterTrack, setFilterTrack] = useState<string>("");
  const [filterRole, setFilterRole] = useState<string>("");
  const [filterChapter, setFilterChapter] = useState<string>("");
  const [filterUntested, setFilterUntested] = useState(false);

  const chapters = useMemo(() => getChapters(), []);
  const tracks = useMemo(() => getTracks(), []);
  const roles = useMemo(() => getRoles(), []);

  const results = useMemo(() => {
    let items = searchConcepts(query);

    if (filterTrack) items = items.filter((c) => c.tracks.includes(filterTrack));
    if (filterRole) items = items.filter((c) => c.role === filterRole);
    if (filterChapter) items = items.filter((c) => c.source_chapters.includes(filterChapter));
    if (filterUntested) items = items.filter((c) => c.question_count === 0);

    items.sort((a, b) => {
      let cmp = 0;
      switch (sortKey) {
        case "name":
          cmp = a.name.localeCompare(b.name);
          break;
        case "tracks":
          cmp = a.tracks.join(",").localeCompare(b.tracks.join(","));
          break;
        case "question_count":
          cmp = a.question_count - b.question_count;
          break;
        case "role":
          cmp = a.role.localeCompare(b.role);
          break;
        case "chapter":
          cmp = (a.source_chapters[0] || "").localeCompare(b.source_chapters[0] || "");
          break;
      }
      return sortDir === "asc" ? cmp : -cmp;
    });

    return items;
  }, [query, sortKey, sortDir, filterTrack, filterRole, filterChapter, filterUntested]);

  const toggleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortDir(sortDir === "asc" ? "desc" : "asc");
    } else {
      setSortKey(key);
      setSortDir("asc");
    }
  };

  const SortIcon = ({ col }: { col: SortKey }) => {
    if (sortKey !== col) return null;
    return sortDir === "asc" ? (
      <ChevronUp className="w-3 h-3 inline ml-0.5" />
    ) : (
      <ChevronDown className="w-3 h-3 inline ml-0.5" />
    );
  };

  return (
    <div className="flex flex-col h-full">
      {/* Search + Filters */}
      <div className="space-y-3 mb-4">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-textTertiary" />
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search concepts..."
            className="w-full pl-9 pr-4 py-2 bg-surface border border-border rounded-lg text-sm text-white placeholder:text-textTertiary focus:outline-none focus:border-accentBlue/50"
          />
        </div>
        <div className="flex flex-wrap gap-2">
          <select
            value={filterTrack}
            onChange={(e) => setFilterTrack(e.target.value)}
            className="text-xs bg-surface border border-border rounded-md px-2 py-1.5 text-textSecondary"
          >
            <option value="">All tracks</option>
            {tracks.map((t) => (
              <option key={t} value={t}>
                {t === "tinyml" ? "TinyML" : t.charAt(0).toUpperCase() + t.slice(1)}
              </option>
            ))}
          </select>
          <select
            value={filterRole}
            onChange={(e) => setFilterRole(e.target.value)}
            className="text-xs bg-surface border border-border rounded-md px-2 py-1.5 text-textSecondary"
          >
            <option value="">All roles</option>
            {roles.map((r) => (
              <option key={r} value={r}>
                {r.charAt(0).toUpperCase() + r.slice(1)}
              </option>
            ))}
          </select>
          <select
            value={filterChapter}
            onChange={(e) => setFilterChapter(e.target.value)}
            className="text-xs bg-surface border border-border rounded-md px-2 py-1.5 text-textSecondary max-w-[200px]"
          >
            <option value="">All chapters</option>
            {chapters.map((ch) => (
              <option key={ch} value={ch}>
                {formatChapter(ch)}
              </option>
            ))}
          </select>
          <label className="flex items-center gap-1.5 text-xs text-textTertiary cursor-pointer">
            <input
              type="checkbox"
              checked={filterUntested}
              onChange={(e) => setFilterUntested(e.target.checked)}
              className="rounded border-border"
            />
            Untested only
          </label>
        </div>
      </div>

      {/* Results count */}
      <div className="text-[10px] font-mono text-textTertiary mb-2 uppercase tracking-wider">
        {results.length} concept{results.length !== 1 ? "s" : ""}
      </div>

      {/* Table */}
      <div className="flex-1 overflow-auto -mx-1">
        <table className="w-full text-xs">
          <thead className="sticky top-0 bg-background z-10">
            <tr className="border-b border-border">
              <th
                onClick={() => toggleSort("name")}
                className="text-left p-2 font-medium text-textTertiary cursor-pointer hover:text-white select-none"
              >
                Name <SortIcon col="name" />
              </th>
              <th
                onClick={() => toggleSort("tracks")}
                className="text-left p-2 font-medium text-textTertiary cursor-pointer hover:text-white select-none w-20"
              >
                Tracks <SortIcon col="tracks" />
              </th>
              <th
                onClick={() => toggleSort("question_count")}
                className="text-right p-2 font-medium text-textTertiary cursor-pointer hover:text-white select-none w-12"
              >
                Qs <SortIcon col="question_count" />
              </th>
              <th
                onClick={() => toggleSort("role")}
                className="text-left p-2 font-medium text-textTertiary cursor-pointer hover:text-white select-none w-24"
              >
                Role <SortIcon col="role" />
              </th>
              <th
                onClick={() => toggleSort("chapter")}
                className="text-left p-2 font-medium text-textTertiary cursor-pointer hover:text-white select-none hidden lg:table-cell"
              >
                Chapter <SortIcon col="chapter" />
              </th>
            </tr>
          </thead>
          <tbody>
            {results.map((c) => (
              <tr
                key={c.id}
                onClick={() => onSelect(c)}
                className={clsx(
                  "border-b border-border/50 cursor-pointer transition-colors",
                  selected === c.id
                    ? "bg-accentBlue/10"
                    : "hover:bg-surface/50"
                )}
              >
                <td className="p-2">
                  <div className="flex items-center gap-1.5">
                    {c.question_count === 0 && (
                      <AlertCircle className="w-3 h-3 text-accentRed shrink-0" />
                    )}
                    <span className="text-white font-medium truncate">
                      {c.name}
                    </span>
                  </div>
                </td>
                <td className="p-2 text-textTertiary">
                  {c.tracks.map((t) => (
                    <span
                      key={t}
                      className="inline-block bg-surface px-1.5 py-0.5 rounded text-[10px] mr-1"
                    >
                      {t}
                    </span>
                  ))}
                </td>
                <td className="p-2 text-right font-mono">
                  <span
                    className={clsx(
                      c.question_count === 0
                        ? "text-accentRed"
                        : c.question_count > 30
                        ? "text-accentAmber"
                        : "text-textSecondary"
                    )}
                  >
                    {c.question_count}
                  </span>
                </td>
                <td className="p-2">
                  <span
                    className={clsx(
                      "inline-block px-1.5 py-0.5 rounded text-[10px] font-medium",
                      c.role === "foundational"
                        ? "bg-accentBlue/20 text-accentBlue"
                        : c.role === "competency"
                        ? "bg-accentGreen/20 text-accentGreen"
                        : "bg-surface text-textTertiary"
                    )}
                  >
                    {c.role}
                  </span>
                </td>
                <td className="p-2 text-textTertiary truncate hidden lg:table-cell max-w-[180px]">
                  {formatChapter(c.source_chapters[0] || "")}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
