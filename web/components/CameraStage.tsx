'use client';

import { useEffect, useRef, useState } from 'react';
import { LandmarkEngine, type CapturedFrame } from '@/lib/mediapipe/landmarks';
import type { PresenceState, PresenceThresholds } from '@/lib/mediapipe/presence';

type Props = {
  onFrame: (frame: CapturedFrame) => void;
  onArrive: () => void;
  onDepart: () => void;
  presence: PresenceState;
  thresholds: PresenceThresholds;
  /**
   * Draw the pose skeleton and the shoulder measurement.
   *
   * This is a calibration instrument, so it stays behind ?tune=1. The HANDS
   * are always drawn: they are what the sign classifier reads, and seeing the
   * joints land on your fingers is how a signer knows they are being tracked
   * before they commit to a phrase.
   */
  showPose: boolean;
};

/*
  Said the way a person would say it. A customer never needs to know that a
  pose model is gating on shoulder width -- only whether the terminal can see
  them well enough to start.
*/
const PRESENCE_LABEL: Record<PresenceState, string> = {
  absent: 'Step closer to begin',
  arriving: 'Almost there…',
  present: 'I can see you',
  leaving: 'Still there?',
};

const PRESENCE_TONE: Record<PresenceState, string> = {
  absent: 'text-ink-faint',
  arriving: 'text-attention',
  present: 'text-action',
  leaving: 'text-attention',
};

/**
 * Pose connections worth drawing: the upper body the classifier actually uses.
 * MediaPipe emits 33 pose points; the legs tell us nothing about signing.
 */
const POSE_EDGES: [number, number][] = [
  [11, 12], // shoulders
  [11, 13],
  [13, 15], // left arm
  [12, 14],
  [14, 16], // right arm
  [11, 23],
  [12, 24],
  [23, 24], // torso
];

/** MediaPipe hand topology: wrist to each fingertip. */
const HAND_EDGES: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [0, 9], [9, 10], [10, 11], [11, 12],
  [0, 13], [13, 14], [14, 15], [15, 16],
  [0, 17], [17, 18], [18, 19], [19, 20],
  [5, 9], [9, 13], [13, 17],
];

export function CameraStage({
  onFrame,
  onArrive,
  onDepart,
  presence,
  thresholds,
  showPose,
}: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const engineRef = useRef<LandmarkEngine | null>(null);
  const [status, setStatus] = useState<'loading' | 'ready' | 'denied' | 'error'>('loading');
  const [message, setMessage] = useState('Loading recognition models');

  // Callbacks change every render; hold them in refs so the camera is not torn
  // down and re-acquired on each one (which makes the webcam LED strobe).
  const cbs = useRef({ onFrame, onArrive, onDepart, showPose });

  useEffect(() => {
    cbs.current = { onFrame, onArrive, onDepart, showPose };
  });

  // Push threshold edits into the running tracker without restarting anything.
  useEffect(() => {
    engineRef.current?.setThresholds(thresholds);
  }, [thresholds]);

  useEffect(() => {
    const engine = new LandmarkEngine();
    engineRef.current = engine;
    let stream: MediaStream | undefined;
    let cancelled = false;

    (async () => {
      try {
        await engine.init();
        if (cancelled) return;

        setMessage('Requesting camera');
        stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 1280, height: 720, facingMode: 'user' },
          audio: false,
        });
        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }

        const video = videoRef.current!;
        video.srcObject = stream;
        await video.play();

        engine.start(video, (frame) => {
          cbs.current.onFrame(frame);
          draw(frame, cbs.current.showPose);
          if (frame.arrived) cbs.current.onArrive();
          if (frame.departed) cbs.current.onDepart();
        });

        setStatus('ready');
      } catch (err) {
        if (cancelled) return;
        const denied = err instanceof DOMException && err.name === 'NotAllowedError';
        setStatus(denied ? 'denied' : 'error');
        setMessage(
          denied
            ? 'Camera access was blocked. Sign language input needs the camera.'
            : `Could not start the camera: ${err instanceof Error ? err.message : String(err)}`,
        );
      }
    })();

    function draw(frame: CapturedFrame, withPose: boolean) {
      const canvas = canvasRef.current;
      const video = videoRef.current;
      if (!canvas || !video || !video.videoWidth) return;

      /*
        The canvas is sized to the box it is DISPLAYED in, not to the camera's
        resolution, and the context is scaled so everything below is in CSS
        pixels. Sizing it 1280x720 and letting CSS shrink it to ~184px meant a
        3px line arrived on screen as less than half a pixel -- the joints were
        being drawn correctly and were simply too small to see.
      */
      const dpr = window.devicePixelRatio || 1;
      const boxW = canvas.clientWidth;
      const boxH = canvas.clientHeight;
      if (!boxW || !boxH) return;

      const cw = Math.round(boxW * dpr);
      const ch = Math.round(boxH * dpr);
      if (canvas.width !== cw || canvas.height !== ch) {
        canvas.width = cw;
        canvas.height = ch;
      }

      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, boxW, boxH);

      /*
        The video is drawn with object-cover, which scales it to fill the box
        and crops the overflow. Landmarks are normalised to the FULL camera
        frame, so they have to go through the same transform -- otherwise the
        skeleton is stretched against a cropped image and floats off the body.
      */
      const scale = Math.max(boxW / video.videoWidth, boxH / video.videoHeight);
      const drawnW = video.videoWidth * scale;
      const drawnH = video.videoHeight * scale;
      const offX = (boxW - drawnW) / 2;
      const offY = (boxH - drawnH) / 2;
      const px = (nx: number) => offX + nx * drawnW;
      const py = (ny: number) => offY + ny * drawnH;

      const { pose, left, right } = frame.overlay;

      const line = (
        pts: { x: number; y: number }[],
        edges: [number, number][],
        color: string,
        lw: number,
      ) => {
        ctx.strokeStyle = color;
        ctx.lineWidth = lw;
        ctx.lineCap = 'round';
        for (const [a, b] of edges) {
          const p = pts[a];
          const q = pts[b];
          if (!p || !q) continue;
          ctx.beginPath();
          ctx.moveTo(px(p.x), py(p.y));
          ctx.lineTo(px(q.x), py(q.y));
          ctx.stroke();
        }
      };

      const dots = (pts: { x: number; y: number }[], color: string, r: number) => {
        ctx.fillStyle = color;
        for (const p of pts) {
          if (!p) continue;
          ctx.beginPath();
          ctx.arc(px(p.x), py(p.y), r, 0, Math.PI * 2);
          ctx.fill();
        }
      };

      if (pose && withPose) {
        const ok = frame.presence === 'present';
        line(pose, POSE_EDGES, ok ? '#0b6e5f' : '#a8500a', 2);
        dots(
          [11, 12, 13, 14, 15, 16, 23, 24, 0].map((i) => pose[i]).filter(Boolean),
          ok ? '#3d9b8a' : '#c77a3a',
          2.5,
        );

        // The shoulder line is the presence signal, so draw it as the measurement.
        const l = pose[11];
        const r = pose[12];
        if (l && r) {
          ctx.strokeStyle = ok ? '#0b6e5f' : '#a8500a';
          ctx.lineWidth = 1;
          ctx.setLineDash([4, 3]);
          ctx.beginPath();
          ctx.moveTo(px(l.x), py(l.y));
          ctx.lineTo(px(r.x), py(r.y));
          ctx.stroke();
          ctx.setLineDash([]);
        }
      }

      /*
        Hands, always. This is the one piece of the overlay a customer benefits
        from: the classifier reads the hands, so joints landing on your fingers
        is live confirmation that it is tracking you. Left and right keep
        distinct colours so a signer can tell which hand dropped out.

        Drawn after the pose so the joints are never buried under the skeleton.
      */
      if (left) {
        line(left, HAND_EDGES, '#1d6fb8', 1.5);
        dots(left, '#4e97d6', 1.6);
      }
      if (right) {
        line(right, HAND_EDGES, '#7b4bb8', 1.5);
        dots(right, '#a279d6', 1.6);
      }
    }

    return () => {
      cancelled = true;
      engine.close();
      engineRef.current = null;
      stream?.getTracks().forEach((t) => t.stop());
    };
  }, []);

  return (
    <div className="flex items-center gap-4">
      {/*
        A self-view, not a stage. Its only job is to tell you that you are in
        frame and being seen -- the same job the small picture of yourself does
        on a video call -- so it is sized accordingly. Earlier versions gave it
        half the screen, which took space from the conversation and the order,
        the two things a customer is actually reading.
      */}
      <div className="relative shrink-0 overflow-hidden rounded-panel border border-line bg-sunk">
        {/*
          Mirrored, so moving your right hand moves the right of the image.
          An unmirrored preview is genuinely disorienting to sign in front of,
          and the overlay is mirrored with it so the skeleton stays on the body.
        */}
        <div className="relative h-[8.5rem] w-[11.5rem] scale-x-[-1] sm:h-[9.5rem] sm:w-[12.75rem]">
          <video ref={videoRef} muted playsInline className="h-full w-full object-cover" />
          <canvas ref={canvasRef} className="pointer-events-none absolute inset-0 h-full w-full" />
        </div>

        {status !== 'ready' && (
          <div className="absolute inset-0 flex items-center justify-center bg-sunk px-3 text-center">
            <p className="text-[0.8125rem] leading-snug text-ink-soft">{message}</p>
          </div>
        )}
      </div>

      <p aria-live="polite" className={`text-[1.0625rem] font-bold ${PRESENCE_TONE[presence]}`}>
        <span
          aria-hidden="true"
          className={[
            'mr-2 inline-block h-2.5 w-2.5 rounded-full align-middle',
            presence === 'present' ? 'bg-action' : presence === 'absent' ? 'bg-line-strong' : 'bg-attention',
          ].join(' ')}
        />
        {PRESENCE_LABEL[presence]}
      </p>
    </div>
  );
}
