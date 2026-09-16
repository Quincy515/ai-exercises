import { useQuery } from '@tanstack/react-query'
import { Link, createFileRoute } from '@tanstack/react-router'

export const Route = createFileRoute('/demo/tanstack-query')({
  component: TanStackQueryDemo,
})

function TanStackQueryDemo() {
  const { data } = useQuery({
    queryKey: ['todos'],
    queryFn: () =>
      Promise.resolve([
        { id: 1, name: 'Alice' },
        { id: 2, name: 'Bob' },
        { id: 3, name: 'Charlie' },
      ]),
    initialData: [],
  })

  return (
    <main className="mx-auto w-full max-w-2xl px-4 py-12">
      <Link
        to="/"
        className="mb-6 inline-block text-primary underline underline-offset-4"
      >
        Back to chat
      </Link>
      <section className="rounded-2xl border bg-card p-6 text-card-foreground sm:p-8">
        <p className="mb-2 text-sm font-medium text-muted-foreground">
          TanStack Query
        </p>
        <h1 className="mb-6 text-3xl font-bold">
          TanStack Query Simple Promise Handling
        </h1>
        <ul className="mb-4 flex flex-col gap-2">
          {data.map((todo) => (
            <li key={todo.id} className="rounded-lg border bg-muted p-3">
              <span className="text-base font-medium">{todo.name}</span>
            </li>
          ))}
        </ul>
      </section>
    </main>
  )
}
