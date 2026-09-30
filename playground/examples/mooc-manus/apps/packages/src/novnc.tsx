import { VNCViewer } from "./components/vnc-viewer";

export default function Novnc() {
  // docker run --rm -d -p 127.0.0.1:8080:3000 -p 127.0.0.1:5900:5900 -p 127.0.0.1:5901:5901 -p 127.0.0.1:9222:9222 --name sandbox-dev sandbox-dev
  return (
    <div className="h-svh w-screen">
      <VNCViewer url="ws://127.0.0.1:5901" viewOnly={false} />
    </div>
  );
}
