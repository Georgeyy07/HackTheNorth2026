import React, { createContext, useContext } from 'react';
import { useImuStreamer } from '../motion/useImuStreamer.js';

const ImuStreamContext = createContext(null);

export function ImuStreamProvider({
  children,
  isNavigating = false,
  tiltAngle = 0,
  maxTiltAngle = 20.0,
  defaultUrl = null,
}) {
  const streamer = useImuStreamer({
    isNavigating,
    tiltAngle,
    maxTiltAngle,
    batchIntervalMs: 200, // 5 times per second
    defaultUrl,
  });

  return (
    <ImuStreamContext.Provider value={streamer}>
      {children}
    </ImuStreamContext.Provider>
  );
}

export function useImuStream() {
  const context = useContext(ImuStreamContext);
  if (!context) {
    throw new Error('useImuStream must be used within an ImuStreamProvider');
  }
  return context;
}
