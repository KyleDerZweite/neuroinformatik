import { NavLink, Outlet } from "react-router-dom";

export function AppShell() {
  return (
    <div className="app-shell">
      <header className="app-header">
        <div>
          <p className="eyebrow">Neuroinformatik</p>
          <h1>Dashboard scaffold</h1>
        </div>
        <nav className="app-nav" aria-label="Primary">
          <NavLink to="/">Train</NavLink>
          <NavLink to="/inspect">Inspect</NavLink>
          <NavLink to="/compare">Compare</NavLink>
        </nav>
      </header>
      <main className="app-main">
        <Outlet />
      </main>
    </div>
  );
}
