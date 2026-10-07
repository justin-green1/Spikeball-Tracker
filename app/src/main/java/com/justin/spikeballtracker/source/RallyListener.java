package com.justin.spikeballtracker.source;

import com.justin.spikeballtracker.record.RallyFootage;
import com.justin.spikeballtracker.tracking.RallyDetector;

/**
 * Told about each rally once the detector has confirmed it, a few seconds
 * after the rally ends (or when the run stops, for one still in progress).
 *
 * <p>Called on the source's own worker thread.
 */
public interface RallyListener {

    /**
     * @param footage the rally's video, from 1 s before it to 3 s after; null
     *                if the source could not capture it
     */
    void onRally(RallyDetector.Rally rally, RallyFootage footage);
}
