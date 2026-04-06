"""
CLI interactivo para BuscoEngino motor de búsqueda.
"""

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm
from rich.table import Table
from rich.text import Text

from search_engine.search.engine import BuscoEngino


def truncate_content(content: str, max_length: int = 150) -> str:
    """Trunca el contenido a una longitud máxima."""
    lines = content.split("\n")
    text = " ".join(line.strip() for line in lines if line.strip())
    if len(text) > max_length:
        return text[:max_length] + "..."
    return text


def display_banner(console: Console) -> None:
    """Muestra el banner del motor."""
    engine_name = Text("BuscoEngino", style="bold cyan")
    subtitle = Text("Motor de búsqueda minimalista", style="dim white")

    panel_content = Text()
    panel_content.append(engine_name)
    panel_content.append("\n")
    panel_content.append(subtitle)

    console.print(Panel(panel_content, expand=False, style="cyan", padding=(1, 2)))


def display_results(console: Console, results: list, max_results: int = 5) -> None:
    """Muestra los resultados en una tabla formateada."""
    if not results:
        console.print("[yellow]No se encontraron resultados[/yellow]")
        return

    table = Table(title="Resultados de búsqueda", show_header=True)
    table.add_column("#", style="dim cyan", width=3)
    table.add_column("Puntaje", style="cyan", width=10)
    table.add_column("Ruta", style="green", width=40)
    table.add_column("Contenido", style="white", width=60)

    for idx, result in enumerate(results[:max_results], start=1):
        content_preview = truncate_content(result.document.content)
        table.add_row(
            str(idx), f"{result.score:.4f}", result.document.path, content_preview
        )

    console.print(table)


def display_top_result(console: Console, result) -> None:
    """Muestra el primer resultado en detalle."""
    if not result:
        return

    panel = Panel(
        Text(result.document.content[:300], style="white"),
        title=f"[cyan]Resultado Top[/cyan] - Puntaje: {result.score:.4f}",
        style="cyan",
        padding=(1, 2),
    )
    console.print(panel)


def main() -> None:
    """Inicia la CLI interactiva del motor de búsqueda."""
    console = Console()

    display_banner(console)

    console.print("[dim]Inicializando motor de búsqueda...[/dim]")
    engine = BuscoEngino()

    doc_count = len(engine.state.documents)
    vocab_size = len(engine.state.vocabulary)
    console.print(
        f"[green]✓[/green] Motor listo | "
        f"[cyan]{doc_count}[/cyan] documentos | "
        f"[cyan]{vocab_size}[/cyan] términos en vocabulario"
    )
    console.print("[dim](Escribe 'exit' para salir)[/dim]\n")

    while True:
        try:
            query = console.input("[cyan]> [/cyan]").strip()

            if not query:
                continue

            if query.lower() in ("exit", "quit", "q"):
                console.print("[cyan]Hasta luego![/cyan]")
                break

            results = engine.search(query)

            if results:
                display_results(console, results)
                console.print()
                display_top_result(console, results[0])

                is_relevant = Confirm.ask(
                    f"\n[cyan]¿Fue relevante el resultado principal ({results[0].document.path})?[/cyan]"
                )
                engine.add_feedback(query, results[0].document.path, is_relevant)
            else:
                console.print("[yellow]No se encontraron resultados[/yellow]")

            console.print()

        except KeyboardInterrupt:
            console.print("\n[red]Búsqueda cancelada[/red]")
            break
        except Exception as e:
            console.print(f"[red]Error: {str(e)}[/red]")


if __name__ == "__main__":
    main()
