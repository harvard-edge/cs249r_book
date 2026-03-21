'use client'
 
export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string }
  reset: () => void
}) {
  return (
    <div className="flex h-screen w-full flex-col items-center justify-center bg-background text-white font-mono">
      <h2 className="text-2xl text-danger mb-4">A critical error occurred.</h2>
      <p className="mb-4 text-primaryDim">{error.message}</p>
      <button
        onClick={() => reset()}
        className="px-4 py-2 bg-primary text-black rounded"
      >
        Try again
      </button>
    </div>
  )
}
