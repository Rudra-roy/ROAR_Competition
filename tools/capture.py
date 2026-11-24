"""Record camera frames of a race, for the README media. Training-time tool: the submission never touches it.

  python tools/capture.py PORT OUTDIR [--every 0.5] [--size 960x540] [--views chase,aerial]

A second CARLA client (like tools/hud.py): waits for the race's car, attaches RGB cameras to it and saves a PNG every
--every seconds of SIMULATED time, plus frames.csv (view, sim time, car x, y, speed) so a still can be picked by where
the car was. The race itself is untouched: start it as usual (tools/eval_slot.sh <slot> <checkout> <label>, server on
port 2000 + 3 * slot) and start this one beside it; it exits when the car is gone. Cameras do not change the score
(the simulation is synchronous), but they cost wall-clock time.
"""
import argparse
import csv
import math
import os
import socket
import threading
import time

import carla

VIEWS = {  # name: (x, z, pitch) relative to the car, metres / degrees
    "chase": (-7.5, 2.6, -6.0),
    "aerial": (-14.0, 22.0, -55.0),
    "nose": (1.9, 1.15, -2.0),
}


def wait_for_car(world, timeout=900):
    t0 = time.time()
    while time.time() - t0 < timeout:
        cars = list(world.get_actors().filter("vehicle.*"))
        if cars:
            hero = [a for a in cars if a.attributes.get("role_name") == "hero"]
            return (hero or cars)[0]
        time.sleep(1)
    raise TimeoutError("no vehicle appeared")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("port", type=int)
    ap.add_argument("outdir")
    ap.add_argument("--every", type=float, default=0.5, help="simulated seconds between frames")
    ap.add_argument("--size", default="960x540")
    ap.add_argument("--views", default="chase,aerial")
    a = ap.parse_args(argv)
    w, h = a.size.split("x")
    os.makedirs(a.outdir, exist_ok=True)
    # Start this BEFORE the server so the standing start is captured: poll the port, never the RPC (a client
    # connecting to a half-started server can hang past its timeout).
    while True:
        try:
            socket.create_connection(("127.0.0.1", a.port), 1).close()
            break
        except OSError:
            time.sleep(1)
    time.sleep(3)
    client = carla.Client("127.0.0.1", a.port)
    client.set_timeout(120.0)
    world = client.get_world()
    car = wait_for_car(world)
    lock, rows, sensors = threading.Lock(), [], []
    for name in a.views.split(","):
        x, z, pitch = VIEWS[name]
        bp = world.get_blueprint_library().find("sensor.camera.rgb")
        for k, v in (("image_size_x", w), ("image_size_y", h), ("fov", "90"), ("sensor_tick", str(a.every))):
            bp.set_attribute(k, v)
        cam = world.spawn_actor(bp, carla.Transform(carla.Location(x=x, z=z), carla.Rotation(pitch=pitch)), attach_to=car)

        def save(img, name=name):
            t = img.timestamp
            img.save_to_disk(os.path.join(a.outdir, "%s_%07.2f.png" % (name, t)))
            tf, v = car.get_transform(), car.get_velocity()
            with lock:
                rows.append((name, t, tf.location.x, tf.location.y, math.hypot(v.x, v.y)))

        cam.listen(save)
        sensors.append(cam)
    print("capturing %s every %.2f s into %s" % (a.views, a.every, a.outdir), flush=True)
    try:
        while True:
            time.sleep(2)
            car.get_transform()  # raises once the car is destroyed or the server is gone
    except RuntimeError:
        pass
    finally:
        for s in sensors:
            try:
                s.stop()
                s.destroy()
            except RuntimeError:
                pass
        with lock, open(os.path.join(a.outdir, "frames.csv"), "w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["view", "t", "x", "y", "speed"])
            wr.writerows(rows)
        print("wrote %d frames" % len(rows), flush=True)


if __name__ == "__main__":
    main()
