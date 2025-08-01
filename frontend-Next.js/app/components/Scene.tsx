'use client';

import { FC } from 'react';
import { useFrame } from '@react-three/fiber';
import { Text } from '@react-three/drei';
import * as THREE from 'three';
import { Sensor } from '../lib/types';
import { useRef } from 'react';

// --- Pre-calculated 3D coordinates for sensors ---
const SENSOR_3D_POSITIONS = [
    [-17, 0.6, -10], [-13, 0.6, 5], [-10, 0.6, -8], [-7, 0.6, 10],
    [-5, 0.6, -2], [-2, 0.6, 8], [0, 0.6, -11], [3, 0.6, 3],
    [5, 0.6, -5], [8, 0.6, 9], [11, 0.6, -9], [14, 0.6, 0],
    [17, 0.6, 7], [19, 0.6, -4], [-18, 0.6, 2], [-15, 0.6, -5],
    [-11, 0.6, 11], [-8, 0.6, -1], [-4, 0.6, 6], [-1, 0.6, -7],
    [2, 0.6, 1], [6, 0.6, -10], [9, 0.6, 5], [12, 0.6, -3],
    [15, 0.6, 10], [18, 0.6, -9], [-19, 0.6, 8], [19, 0.6, 11]
];

// --- 3D Sensor Component ---
const SensorNode: FC<{ id: number; isOff: boolean; position: [number, number, number] }> = ({ id, isOff, position }) => {
    const onColor = '#3b82f6';
    const offColor = '#4b5563';
    const emissiveIntensity = isOff ? 0 : 2.5;

    return (
        <mesh position={position} castShadow>
            <sphereGeometry args={[0.7, 32, 32]} />
            <meshStandardMaterial
                color={isOff ? offColor : onColor}
                emissive={isOff ? offColor : onColor}
                emissiveIntensity={emissiveIntensity}
                roughness={0.4}
                metalness={0.2}
            />
            <Text position={[0, 1.1, 0]} color="white" fontSize={0.6} anchorX="center" anchorY="middle" outlineWidth={0.02} outlineColor="black" >
                {id + 1}
            </Text>
        </mesh>
    );
};

// --- NEW COMPONENT: This contains the rotating group and the useFrame hook ---
export const RotatingSceneContent: FC<{ sensors: Sensor[]; groundTexture: THREE.CanvasTexture | null }> = ({ sensors, groundTexture }) => {
    const sceneGroupRef = useRef<THREE.Group>(null);

    // This hook will run on every rendered frame
    useFrame((state, delta) => {
        if (sceneGroupRef.current) {
            // Slowly rotate the entire scene
            sceneGroupRef.current.rotation.y += delta * 0.05;
        }
    });

    return (
        <group ref={sceneGroupRef} position={[0, -1.0, 0]}>
             <group position={[0, -0.5, 0]}>
                <mesh rotation={[-Math.PI / 2, 0, 0]} receiveShadow>
                    <planeGeometry args={[45, 30]} />
                    <meshStandardMaterial map={groundTexture} color={!groundTexture ? '#166534' : undefined} roughness={1} />
                </mesh>
                <mesh position={[0, -1, 15]}><boxGeometry args={[45, 2, 0.1]} /><meshStandardMaterial color="#8d6e63" /></mesh>
                <mesh position={[0, -1, -15]}><boxGeometry args={[45, 2, 0.1]} /><meshStandardMaterial color="#8d6e63" /></mesh>
                <mesh position={[22.5, -1, 0]} rotation={[0, -Math.PI / 2, 0]}><boxGeometry args={[30, 2, 0.1]} /><meshStandardMaterial color="#8d6e63" /></mesh>
                <mesh position={[-22.5, -1, 0]} rotation={[0, Math.PI / 2, 0]}><boxGeometry args={[30, 2, 0.1]} /><meshStandardMaterial color="#8d6e63" /></mesh>
                <mesh position={[0, -2, 0]} rotation={[-Math.PI / 2, 0, 0]}><planeGeometry args={[45, 30]} /><meshStandardMaterial color="#5d4037" /></mesh>
            </group>
            {sensors?.map((sensor) => (
                <SensorNode key={sensor.id} id={sensor.id} isOff={sensor.is_off} position={SENSOR_3D_POSITIONS[sensor.id] as [number, number, number]} />
            ))}
        </group>
    );
}
