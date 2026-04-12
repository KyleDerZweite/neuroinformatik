import { createBrowserRouter } from "react-router-dom";

import { AppShell } from "../components/AppShell";
import { ComparePage } from "../pages/ComparePage";
import { InspectPage } from "../pages/InspectPage";
import { TrainPage } from "../pages/TrainPage";

export const router = createBrowserRouter([
  {
    path: "/",
    element: <AppShell />,
    children: [
      {
        index: true,
        element: <TrainPage />,
      },
      {
        path: "inspect",
        element: <InspectPage />,
      },
      {
        path: "compare",
        element: <ComparePage />,
      },
    ],
  },
]);
