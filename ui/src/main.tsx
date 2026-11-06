import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import Post from "./post/Post";
import "./styles.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <Post />
  </StrictMode>,
);
