/**
 * Fleet Video Synchronization Helper
 * Implements the developer handoff specification for sessions 2–5.
 */

export function resolveFleetVideo(sync, variant, carID, mapTimestampMs) {
  const scenarioKey = (variant || 'staggered').toLowerCase();
  const scenarioConfig = sync[scenarioKey] || sync['staggered'];
  const carCfg = scenarioConfig ? scenarioConfig[carID] : null;

  if (!carCfg) {
    return {
      state: 'waiting',
      url: null,
      currentTime: 0,
      message: `Car ${carID} not found in scenario ${scenarioKey}`
    };
  }

  const baseStartMs = Date.parse(sync.start || '2026-09-20T12:00:00Z');
  const launchDelayMs = (carCfg.launchDelaySeconds || 0) * 1000;
  const launchTimestampMs = baseStartMs + launchDelayMs;

  if (mapTimestampMs < launchTimestampMs) {
    const waitSec = ((launchTimestampMs - mapTimestampMs) / 1000).toFixed(1);
    return {
      state: 'waiting',
      url: null,
      currentTime: 0,
      message: `Launches in ${waitSec}s (${carCfg.firstSession})`
    };
  }

  const sessionOrder = ['session2', 'session3', 'session4', 'session5'];
  const sessionDurations = {
    session2: 482.0,
    session3: 245.0,
    session4: 310.0,
    session5: 140.0,
  };
  const gaps = {
    session2: 1.610,
    session3: 5.615,
    session4: 3.919,
  };
  const startOffsets = {
    session2: 8.902,
    session3: 0.118,
    session4: 0.043,
    session5: 0.294,
  };

  const firstSession = carCfg.firstSession || 'session2';
  const startIndex = sessionOrder.indexOf(firstSession);
  const initialOffset = carCfg.videoAtLaunchSeconds !== undefined ? carCfg.videoAtLaunchSeconds : (startOffsets[firstSession] || 0);

  let elapsedSinceLaunchSec = (mapTimestampMs - launchTimestampMs) / 1000.0;

  for (let i = startIndex; i < sessionOrder.length; i++) {
    const sessName = sessionOrder[i];
    const dur = sessionDurations[sessName];
    const offset = (i === startIndex) ? initialOffset : 0;
    const remainingInSession = dur - offset;

    if (elapsedSinceLaunchSec <= remainingInSession) {
      const currentVideoTime = offset + elapsedSinceLaunchSec;
      const url = `/demo_view/videos/${sessName}.mp4`;
      return {
        state: 'playing',
        session: sessName,
        url: url,
        currentTime: Math.max(0, currentVideoTime),
        duration: dur,
        message: `${sessName.toUpperCase()} @ ${currentVideoTime.toFixed(1)}s`
      };
    }

    elapsedSinceLaunchSec -= remainingInSession;

    const gapSec = gaps[sessName] || 0;
    if (elapsedSinceLaunchSec < gapSec) {
      return {
        state: 'recording_gap',
        session: sessName,
        url: null,
        currentTime: 0,
        message: `Recording Gap (${gapSec.toFixed(1)}s between ${sessName} and ${sessionOrder[i+1]})`
      };
    }

    elapsedSinceLaunchSec -= gapSec;
  }

  return {
    state: 'finished',
    url: null,
    currentTime: 0,
    message: 'Fleet Patrol Mission Completed'
  };
}
