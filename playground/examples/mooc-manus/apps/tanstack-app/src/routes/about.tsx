import { Link, createFileRoute } from '@tanstack/react-router'

export const Route = createFileRoute('/about')({
  component: About,
})

function About() {
  return (
    <main className="mx-auto w-full max-w-5xl px-4 py-12">
      <Link
        to="/"
        className="mb-6 inline-block text-primary underline underline-offset-4"
      >
        Back to chat
      </Link>
      <section className="rounded-2xl border bg-card p-6 text-card-foreground sm:p-8">
        <p className="mb-2 text-sm font-medium text-muted-foreground">About</p>
        <h1 className="mb-3 text-4xl font-bold sm:text-5xl">
          A small starter with room to grow.
        </h1>
        <p className="m-0 max-w-3xl text-base leading-8 text-muted-foreground">
          TanStack Start gives you type-safe routing, server functions, and
          modern SSR defaults. Use this as a clean foundation, then layer in
          your own routes, styling, and add-ons.
        </p>
      </section>
    </main>
  )
}
