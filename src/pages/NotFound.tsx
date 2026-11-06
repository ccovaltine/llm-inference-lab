import { Link } from "react-router";
import "../App.css";

export default function NotFound() {
  return (
    <main className="notfound-page">
      <div className="notfound-card">
        <p className="notfound-code" aria-hidden="true">404</p>
        <h1>Page not found</h1>
        <Link className="button primary" to="/">Back to Home</Link>
      </div>
    </main>
  );
}
