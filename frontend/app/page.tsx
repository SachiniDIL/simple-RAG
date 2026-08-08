"use client";

import { useMutation } from "@tanstack/react-query";
import { FormEvent, useState } from "react";

const API_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

type QueryResult = {
  text: string;
  source: string;
  chunk_index: number;
  score: number;
  rrf_score: number | null;
};

type QueryResponse = {
  query: string;
  mode: "hybrid" | "semantic-only";
  rejected: boolean;
  message: string | null;
  results: QueryResult[];
  answer: string | null;
};

async function postQuery(body: {
  query: string;
  hybrid: boolean;
  generate: boolean;
}): Promise<QueryResponse> {
  const res = await fetch(`${API_URL}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    throw new Error(`Request failed with status ${res.status}`);
  }

  return res.json();
}

export default function Home() {
  const [question, setQuestion] = useState("");
  const [hybrid, setHybrid] = useState(false);
  const [generate, setGenerate] = useState(false);

  const mutation = useMutation({
    mutationFn: postQuery,
  });

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (!question.trim() || mutation.isPending) return;
    mutation.mutate({ query: question.trim(), hybrid, generate });
  }

  const data = mutation.data;

  return (
    <div className="flex flex-1 justify-center bg-zinc-50 px-4 py-10 dark:bg-black sm:px-8">
      <main className="flex w-full max-w-2xl flex-col gap-6">
        <header>
          <h1 className="text-2xl font-semibold tracking-tight text-black dark:text-zinc-50">
            Simple RAG
          </h1>
          <p className="mt-1 text-sm text-zinc-600 dark:text-zinc-400">
            Ask a question and retrieve relevant chunks from the indexed
            documents.
          </p>
        </header>

        <form onSubmit={handleSubmit} className="flex flex-col gap-3">
          <div className="flex gap-2">
            <input
              type="text"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Ask a question..."
              className="flex-1 rounded-md border border-zinc-300 bg-white px-3 py-2 text-sm text-black outline-none focus:border-zinc-500 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-50 dark:focus:border-zinc-500"
            />
            <button
              type="submit"
              disabled={!question.trim() || mutation.isPending}
              className="rounded-md bg-black px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-zinc-800 disabled:cursor-not-allowed disabled:opacity-40 dark:bg-zinc-50 dark:text-black dark:hover:bg-zinc-200"
            >
              {mutation.isPending ? "Asking..." : "Ask"}
            </button>
          </div>

          <div className="flex gap-6 text-sm text-zinc-700 dark:text-zinc-300">
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={hybrid}
                onChange={(e) => setHybrid(e.target.checked)}
                className="h-4 w-4 rounded border-zinc-300 dark:border-zinc-700"
              />
              Hybrid search
            </label>
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={generate}
                onChange={(e) => setGenerate(e.target.checked)}
                className="h-4 w-4 rounded border-zinc-300 dark:border-zinc-700"
              />
              Generate answer
            </label>
          </div>
        </form>

        {mutation.isPending && (
          <div className="rounded-md border border-zinc-200 bg-white px-4 py-3 text-sm text-zinc-600 dark:border-zinc-800 dark:bg-zinc-900 dark:text-zinc-400">
            {generate
              ? "Retrieving chunks and generating an answer, this can take a while..."
              : "Retrieving relevant chunks..."}
          </div>
        )}

        {mutation.isError && (
          <div className="rounded-md border border-red-300 bg-red-50 px-4 py-3 text-sm text-red-800 dark:border-red-900 dark:bg-red-950 dark:text-red-300">
            <p className="font-medium">Couldn&apos;t reach the backend.</p>
            <p className="mt-1 text-red-700 dark:text-red-400">
              {mutation.error instanceof Error
                ? mutation.error.message
                : "Unknown error"}{" "}
              &mdash; is the API running at {API_URL}?
            </p>
          </div>
        )}

        {data && data.rejected && (
          <div className="rounded-md border border-zinc-200 bg-zinc-100 px-4 py-3 text-sm text-zinc-700 dark:border-zinc-800 dark:bg-zinc-900 dark:text-zinc-300">
            {data.message ?? "No relevant information found."}
          </div>
        )}

        {data && !data.rejected && (
          <div className="flex flex-col gap-4">
            {data.answer && (
              <div className="rounded-md border border-blue-200 bg-blue-50 px-4 py-3 dark:border-blue-900 dark:bg-blue-950">
                <p className="text-xs font-semibold uppercase tracking-wide text-blue-700 dark:text-blue-400">
                  Generated answer
                </p>
                <p className="mt-1 whitespace-pre-wrap text-sm text-blue-950 dark:text-blue-100">
                  {data.answer}
                </p>
              </div>
            )}

            <div className="flex flex-col gap-3">
              <p className="text-xs font-semibold uppercase tracking-wide text-zinc-500 dark:text-zinc-500">
                Retrieved chunks ({data.results.length})
              </p>
              {data.results.map((result, i) => (
                <div
                  key={`${result.source}-${result.chunk_index}-${i}`}
                  className="rounded-md border border-zinc-200 bg-white px-4 py-3 dark:border-zinc-800 dark:bg-zinc-900"
                >
                  <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-zinc-500 dark:text-zinc-400">
                    <span className="font-medium text-zinc-700 dark:text-zinc-300">
                      {result.source}
                    </span>
                    <span>chunk {result.chunk_index}</span>
                    <span>score {result.score.toFixed(4)}</span>
                    {result.rrf_score !== null && (
                      <span>rrf {result.rrf_score.toFixed(4)}</span>
                    )}
                  </div>
                  <p className="mt-2 whitespace-pre-wrap text-sm text-zinc-800 dark:text-zinc-200">
                    {result.text}
                  </p>
                </div>
              ))}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
