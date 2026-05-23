'use client';

import { FC, useRef } from 'react';
import { ThreeEvent, useFrame } from '@react-three/fiber';
import { Text } from '@react-three/drei';
import * as THREE from 'three';
import { Sensor, SensorDetail } from '../lib/types';
import { SENSOR_3D_POSITIONS } from '../lib/constants';

const GATEWAY_POSITION: [number, number, number] = [0, 1.0, 0];

const SensorNode: FC<{
    id: number;
    isOff: boolean;
    isAnomaly?: boolean;
    position: [number, number, number];
    selected?: boolean;
    onSelect?: (id: number) => void;
}> = ({ id, isOff, isAnomaly = false, position, selected = false, onSelect }) => {
    const meshRef = useRef<THREE.Mesh>(null);
    const onColor = isAnomaly ? '#ef4444' : '#3b82f6';
    const offColor = isAnomaly ? '#7f1d1d' : '#4b5563';
    const anomalyRingColor = '#ef4444';
    const emissiveIntensity = isOff ? 0 : 2.5;
    const handleClick = (event: ThreeEvent<MouseEvent>) => {
        event.stopPropagation();
        onSelect?.(id);
    };

    useFrame(({ clock }) => {
        if (!meshRef.current) return;
        if (!isAnomaly) {
            meshRef.current.scale.setScalar(selected ? 1.22 : 1);
            return;
        }
        const pulse = 1 + Math.sin(clock.elapsedTime * 5 + id) * 0.08;
        meshRef.current.scale.setScalar((selected ? 1.22 : 1) * pulse);
    });

    return (
        <mesh
            ref={meshRef}
            position={position}
            castShadow
            onClick={handleClick}
            onPointerOver={(event) => {
                event.stopPropagation();
                document.body.style.cursor = 'pointer';
            }}
            onPointerOut={() => {
                document.body.style.cursor = 'default';
            }}
        >
            <sphereGeometry args={[0.7, 32, 32]} />
            <meshStandardMaterial
                color={selected ? '#e0f2fe' : isOff ? offColor : onColor}
                emissive={selected ? '#38bdf8' : isOff ? (isAnomaly ? '#ef4444' : offColor) : onColor}
                emissiveIntensity={selected ? 2.8 : isAnomaly ? (isOff ? 0.75 : 2.8) : emissiveIntensity}
                roughness={0.4}
                metalness={0.2}
            />
            {isAnomaly && (
                <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.72, 0]}>
                    <ringGeometry args={[1.0, 1.28, 48]} />
                    <meshBasicMaterial color={anomalyRingColor} transparent opacity={isOff ? 0.55 : 0.7} />
                </mesh>
            )}
            <Text
                position={[0, 1.1, 0]}
                color="white"
                fontSize={0.6}
                anchorX="center"
                anchorY="middle"
                outlineWidth={0.02}
                outlineColor="black"
            >
                {id + 1}
            </Text>
        </mesh>
    );
};

const GatewayNode: FC = () => {
    const headRef = useRef<THREE.Group>(null);

    useFrame(({ clock }) => {
        if (headRef.current) {
            headRef.current.rotation.y = clock.elapsedTime * 0.25;
        }
    });

    return (
        <group position={GATEWAY_POSITION}>
            <mesh position={[0, 1.15, 0]} castShadow>
                <cylinderGeometry args={[0.08, 0.12, 3.4, 16]} />
                <meshStandardMaterial color="#94a3b8" roughness={0.35} metalness={0.45} />
            </mesh>
            <group ref={headRef} position={[0, 3.0, 0]}>
                <mesh castShadow>
                    <octahedronGeometry args={[0.78, 1]} />
                    <meshStandardMaterial color="#facc15" emissive="#f59e0b" emissiveIntensity={1.25} roughness={0.25} metalness={0.3} />
                </mesh>
            </group>
            <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.45, 0]}>
                <ringGeometry args={[1.25, 1.45, 64]} />
                <meshBasicMaterial color="#facc15" transparent opacity={0.26} />
            </mesh>
            <Text
                position={[0, 4.0, 0]}
                color="white"
                fontSize={0.5}
                anchorX="center"
                anchorY="middle"
                outlineWidth={0.02}
                outlineColor="black"
            >
                FARM GATEWAY
            </Text>
        </group>
    );
};

export const RotatingSceneContent: FC<{
    sensors: Sensor[];
    sensorDetails?: SensorDetail[];
    groundTexture: THREE.CanvasTexture | null;
    selectedSensorId?: number | null;
    onSelectSensor?: (id: number) => void;
}> = ({ sensors, sensorDetails = [], groundTexture, selectedSensorId = null, onSelectSensor }) => {
    const sceneGroupRef = useRef<THREE.Group>(null);
    const detailById = new Map(sensorDetails.map(detail => [detail.id, detail]));

    useFrame((state, delta) => {
        if (sceneGroupRef.current) {
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
                <mesh position={[0, -1, 15]}>
                    <boxGeometry args={[45, 2, 0.1]} />
                    <meshStandardMaterial color="#8d6e63" />
                </mesh>
                <mesh position={[0, -1, -15]}>
                    <boxGeometry args={[45, 2, 0.1]} />
                    <meshStandardMaterial color="#8d6e63" />
                </mesh>
                <mesh position={[22.5, -1, 0]} rotation={[0, -Math.PI / 2, 0]}>
                    <boxGeometry args={[30, 2, 0.1]} />
                    <meshStandardMaterial color="#8d6e63" />
                </mesh>
                <mesh position={[-22.5, -1, 0]} rotation={[0, Math.PI / 2, 0]}>
                    <boxGeometry args={[30, 2, 0.1]} />
                    <meshStandardMaterial color="#8d6e63" />
                </mesh>
                <mesh position={[0, -2, 0]} rotation={[-Math.PI / 2, 0, 0]}>
                    <planeGeometry args={[45, 30]} />
                    <meshStandardMaterial color="#5d4037" />
                </mesh>
            </group>
            <GatewayNode />
            {sensors?.filter(sensor => !sensor.is_off && SENSOR_3D_POSITIONS[sensor.id]).map(sensor => {
                const position = SENSOR_3D_POSITIONS[sensor.id] as [number, number, number];
                return (
                    <line key={`trace-${sensor.id}`}>
                        <bufferGeometry>
                            <bufferAttribute
                                attach="attributes-position"
                                args={[new Float32Array([
                                    GATEWAY_POSITION[0], GATEWAY_POSITION[1], GATEWAY_POSITION[2],
                                    position[0], position[1], position[2],
                                ]), 3]}
                            />
                        </bufferGeometry>
                        <lineBasicMaterial color="#38bdf8" transparent opacity={0.18} />
                    </line>
                );
            })}
            {sensors?.map(sensor => (
                <SensorNode
                    key={sensor.id}
                    id={sensor.id}
                    isOff={sensor.is_off}
                    isAnomaly={detailById.get(sensor.id)?.is_anomaly || false}
                    position={SENSOR_3D_POSITIONS[sensor.id] as [number, number, number]}
                    selected={selectedSensorId === sensor.id}
                    onSelect={onSelectSensor}
                />
            ))}
        </group>
    );
};
