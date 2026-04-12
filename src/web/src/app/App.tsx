import { RouterProvider } from "react-router-dom";

import { SessionProvider } from "../lib/session/SessionProvider";
import { router } from "./router";

export function App() {
  return (
    <SessionProvider>
      <RouterProvider router={router} />
    </SessionProvider>
  );
}
