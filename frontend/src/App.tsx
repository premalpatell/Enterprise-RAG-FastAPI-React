import { useState } from 'react'

const API = 'http://localhost:8000'

type Turn = { question: string; answer: string; context: string }

function App() {
  // --- Upload state ---
  const [file, setFile] = useState<File | null>(null)
  const [isUploading, setIsUploading] = useState(false)
  const [uploadMessage, setUploadMessage] = useState('')
  const [chunkCount, setChunkCount] = useState<number | null>(null)

  // --- Chat state ---
  const [question, setQuestion] = useState('')
  const [isAsking, setIsAsking] = useState(false)
  const [turns, setTurns] = useState<Turn[]>([])
  const [chatError, setChatError] = useState('')
  const [openContext, setOpenContext] = useState<number | null>(null)

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) {
      setFile(e.target.files[0])
      setUploadMessage('')
    }
  }

  const handleUpload = async () => {
    if (!file) return
    setIsUploading(true)
    setUploadMessage('Processing document — embedding sentences in batches, this takes a minute...')

    const formData = new FormData()
    formData.append('file', file)

    try {
      const res = await fetch(`${API}/api/documents/upload`, { method: 'POST', body: formData })
      const data = await res.json()
      if (res.ok && !data.error) {
        setChunkCount(data.semantic_chunks_created)
        setUploadMessage(`Indexed ${data.filename} into ${data.semantic_chunks_created} semantic chunks.`)
        setTurns([])
      } else {
        setUploadMessage(`Error: ${data.error ?? 'upload failed'}`)
      }
    } catch {
      setUploadMessage('Network error. Is the FastAPI server running on port 8000?')
    } finally {
      setIsUploading(false)
    }
  }

  const handleAsk = async (e: React.FormEvent) => {
    e.preventDefault()
    const q = question.trim()
    if (!q || isAsking) return

    setIsAsking(true)
    setChatError('')
    setQuestion('')

    try {
      const res = await fetch(`${API}/api/documents/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: q }),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail ?? 'query failed')
      setTurns((t) => [...t, { question: q, answer: data.answer, context: data.context_used ?? '' }])
    } catch (err) {
      setChatError(
        err instanceof Error && err.message !== 'Failed to fetch'
          ? err.message
          : 'Network error, or no document has been indexed yet.',
      )
    } finally {
      setIsAsking(false)
    }
  }

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900">
      <div className="mx-auto max-w-3xl px-4 py-10 space-y-6">

        <header className="space-y-2">
          <h1 className="text-3xl font-bold tracking-tight">Semantic RAG Engine</h1>
          <p className="text-sm text-slate-500">
            Chunk boundaries are placed where adjacent-sentence cosine similarity falls into the
            bottom decile of the document&rsquo;s own distribution — not at a fixed character count.
          </p>
        </header>

        {/* --- Upload --- */}
        <section className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
          <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-500">
            1 · Index a document
          </h2>
          <div className="flex flex-wrap items-center gap-3">
            <input
              type="file"
              accept=".pdf"
              onChange={handleFileChange}
              className="text-sm text-slate-600 file:mr-3 file:rounded-md file:border-0 file:bg-slate-900 file:px-4 file:py-2 file:text-sm file:font-medium file:text-white hover:file:bg-slate-700"
            />
            <button
              onClick={handleUpload}
              disabled={!file || isUploading}
              className="rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-40"
            >
              {isUploading ? 'Building index…' : 'Process document'}
            </button>
          </div>
          {uploadMessage && (
            <p
              className={`mt-3 text-sm ${
                uploadMessage.startsWith('Error') || uploadMessage.startsWith('Network')
                  ? 'text-red-600'
                  : 'text-emerald-700'
              }`}
            >
              {uploadMessage}
            </p>
          )}
        </section>

        {/* --- Chat --- */}
        <section className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
          <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-500">
            2 · Ask a question
          </h2>

          <form onSubmit={handleAsk} className="flex gap-2">
            <input
              type="text"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder={chunkCount ? 'Who is the main character?' : 'Index a document first…'}
              className="flex-1 rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none"
            />
            <button
              type="submit"
              disabled={isAsking || !question.trim()}
              className="rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-40"
            >
              {isAsking ? 'Thinking…' : 'Ask'}
            </button>
          </form>

          {chatError && <p className="mt-3 text-sm text-red-600">{chatError}</p>}

          {turns.length === 0 && !chatError && (
            <p className="mt-4 text-sm text-slate-400">No questions yet.</p>
          )}

          <div className="mt-5 space-y-5">
            {turns.map((t, i) => (
              <div key={i} className="border-t border-slate-100 pt-4 first:border-0 first:pt-0">
                <p className="text-sm font-medium text-slate-900">{t.question}</p>
                <p className="mt-2 whitespace-pre-wrap text-sm leading-relaxed text-slate-700">
                  {t.answer}
                </p>
                {t.context && (
                  <>
                    <button
                      onClick={() => setOpenContext(openContext === i ? null : i)}
                      className="mt-2 text-xs font-medium text-blue-600 hover:underline"
                    >
                      {openContext === i ? 'Hide' : 'Show'} retrieved context (
                      {t.context.length.toLocaleString()} chars)
                    </button>
                    {openContext === i && (
                      <pre className="mt-2 max-h-64 overflow-auto rounded-md bg-slate-50 p-3 text-xs whitespace-pre-wrap text-slate-600">
                        {t.context}
                      </pre>
                    )}
                  </>
                )}
              </div>
            ))}
          </div>
        </section>

      </div>
    </div>
  )
}

export default App
