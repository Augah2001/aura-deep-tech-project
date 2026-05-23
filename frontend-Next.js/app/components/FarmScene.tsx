'use client';

import { FC, memo, useEffect, useState } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';
import { Sensor, SensorDetail } from '../lib/types';
import { RotatingSceneContent } from './Scene';

const FarmScene: FC<{
    sensors: Sensor[];
    sensorDetails?: SensorDetail[];
    selectedSensorId?: number | null;
    onSelectSensor?: (id: number) => void;
}> = memo(({ sensors, sensorDetails = [], selectedSensorId = null, onSelectSensor }) => {
    const [groundTexture, setGroundTexture] = useState<THREE.CanvasTexture | null>(null);

    useEffect(() => {
        const canvas = document.createElement('canvas');
        canvas.width = 512;
        canvas.height = 512;
        const context = canvas.getContext('2d');
        if (!context) return;

        const grassGradient = context.createLinearGradient(0, 0, 0, 512);
        grassGradient.addColorStop(0, '#166534');
        grassGradient.addColorStop(1, '#14532d');
        context.fillStyle = grassGradient;
        context.fillRect(0, 0, 512, 512);

        for (let i = 0; i < 10000; i++) {
            context.fillStyle = `rgba(0,0,0,${Math.random() * 0.1})`;
            context.fillRect(Math.random() * 512, Math.random() * 512, 2, 2);
        }

        const mudPatches = [
            { x: 0.2, y: 0.3, r: 0.2 },
            { x: 0.7, y: 0.8, r: 0.25 },
            { x: 0.8, y: 0.2, r: 0.15 },
        ];
        mudPatches.forEach(patch => {
            const mudGradient = context.createRadialGradient(
                patch.x * 512,
                patch.y * 512,
                0,
                patch.x * 512,
                patch.y * 512,
                patch.r * 512,
            );
            mudGradient.addColorStop(0, 'rgba(93, 64, 55, 0.7)');
            mudGradient.addColorStop(0.8, 'rgba(121, 85, 72, 0.4)');
            mudGradient.addColorStop(1, 'rgba(121, 85, 72, 0)');
            context.fillStyle = mudGradient;
            context.fillRect(0, 0, 512, 512);
        });

        const texture = new THREE.CanvasTexture(canvas);
        texture.needsUpdate = true;
        setGroundTexture(texture);
    }, []);

    return (
        <Canvas shadows camera={{ position: [-10, 20, 30], fov: 50 }} gl={{ alpha: false }} flat={true}>
            <color attach="background" args={['#87B7D8']} />
            <fog attach="fog" args={['#C9E2C3', 48, 95]} />
            <ambientLight intensity={0.95} />
            <hemisphereLight args={['#E0F2FE', '#8D6E63', 1.25]} />
            <directionalLight
                position={[25, 30, 15]}
                intensity={3.0}
                color="#FFF3B0"
                castShadow
                shadow-mapSize-width={2048}
                shadow-mapSize-height={2048}
                shadow-camera-far={80}
                shadow-camera-left={-25}
                shadow-camera-right={25}
                shadow-camera-top={25}
                shadow-camera-bottom={-25}
            />
            <mesh position={[18, 20, -20]}>
                <sphereGeometry args={[2.1, 40, 40]} />
                <meshBasicMaterial color="#FDE68A" />
            </mesh>
            <RotatingSceneContent
                sensors={sensors}
                sensorDetails={sensorDetails}
                groundTexture={groundTexture}
                selectedSensorId={selectedSensorId}
                onSelectSensor={onSelectSensor}
            />
            <OrbitControls enablePan={true} enableZoom={true} minPolarAngle={Math.PI / 8} maxPolarAngle={Math.PI / 2.1} minDistance={15} maxDistance={70} />
        </Canvas>
    );
});

FarmScene.displayName = 'FarmScene';

export default FarmScene;
