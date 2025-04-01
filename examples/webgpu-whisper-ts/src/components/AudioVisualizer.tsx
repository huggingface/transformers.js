import { useCallback, useEffect, useRef } from "react";

interface AudioVisualizerProps {
  stream: MediaStream;
  props:[]
}

export default function AudioVisualizer({ stream, ...props }: AudioVisualizerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const animationFrameRef = useRef<number>(0);

  const visualize = useCallback((stream: MediaStream) => {
    if (!stream) return;

    const cleanup = () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      audioContextRef.current?.close();
      console.debug("Audio context cleaned up");
    };

    try {
      audioContextRef.current = new AudioContext();
      const source = audioContextRef.current.createMediaStreamSource(stream);
      const analyser = audioContextRef.current.createAnalyser();
      analyser.fftSize = 2048;
      source.connect(analyser);

      const canvas = canvasRef.current;
      if (!canvas) return;

      const canvasCTX = canvas.getContext("2d");
      if (!canvasCTX) {
        throw new Error("Failed to get canvas context");
      }

      const bufferLength = analyser.frequencyBinCount;
      const arrayData = new Uint8Array(bufferLength);

      const drawVisual = () => {
        try {
          analyser.getByteTimeDomainData(arrayData);

          canvasCTX.fillStyle = "#ffffff";
          canvasCTX.fillRect(0, 0, canvas.width, canvas.height);

          canvasCTX.beginPath();
          const sliceWidth = (canvas.width * 1.0) / bufferLength;
          let x = 0;

          for (let i = 0; i < bufferLength; i++) {
            const v = arrayData[i] / 128.0;
            const y = (v * canvas.height) / 2;

            if (i === 0) {
              canvasCTX.moveTo(x, y);
            } else {
              canvasCTX.lineTo(x, y);
            }

            x += sliceWidth;
          }

          canvasCTX.lineTo(canvas.width, canvas.height / 2);
          canvasCTX.strokeStyle = "rgba(70, 24, 255, 1)";
          canvasCTX.lineWidth = 2;
          canvasCTX.stroke();

          animationFrameRef.current = requestAnimationFrame(drawVisual);
        } catch (drawError) {
          console.error("Drawing error:", drawError);
          cleanup();
        }
      };

      drawVisual();
      console.debug("Started visualization");

      return () => {
        source?.disconnect();
        analyser?.disconnect();
        cleanup();
      };
    } catch (error) {
      console.error("Visualization setup failed:", error);
      cleanup();
      return () => {};
    }
  }, []);

  useEffect(() => {
    if (!stream) return;

    const cleanup = visualize(stream);
    return () => {
      console.debug("Cleaning up visualizer");
      cleanup?.();
    };
  }, [visualize, stream]);

  return <canvas ref={canvasRef} {...props} width={720} height={240} />;
}