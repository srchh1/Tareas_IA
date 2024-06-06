# Documentación del Código del Juego

## Descripción General
El código describe un juego desarrollado con la biblioteca Phaser, que utiliza una arquitectura de red neuronal para tomar decisiones automáticas dentro del juego. El entorno del juego se compone de un protagonista, enemigos y proyectiles, gestionados con físicas y animaciones.

## Configuración Inicial
- **Dimensiones**: Se establece un área de juego de 800x400.
- **Variables de Estado**: Se manejan estados para controlar el movimiento y acciones de los objetos dentro del juego, así como para controlar el flujo del juego (pausa, reinicio).
- **Red Neuronal**: Se utilizan tres redes neuronales para manejar diferentes aspectos del comportamiento automático del juego.

## Precarga (`preload`)
Se cargan los recursos gráficos como imágenes y spritesheets necesarios para los objetos del juego, incluyendo el fondo, personaje, enemigos y proyectiles.

```javascript
juego.load.image('fondo', 'assets/game/castle.jpg');
juego.load.spritesheet('personaje', 'assets/sprites/castlevania2.png', 24, 42);
juego.load.image('enemigo', 'assets/game/elmeromero.png');
juego.load.image('proyectil', 'assets/sprites/bola.png');
juego.load.image('menu', 'assets/game/menu.png');
juego.load.image('muerto', 'assets/game/castieso.png', 1, 1);
```

## Creación (create)

Se configura el entorno físico del juego y se inicializan los objetos del juego (fondo, personaje, enemigos, proyectiles) con sus propiedades físicas y gráficas adecuadas.

```javascript
juego.physics.startSystem(Phaser.Physics.ARCADE);
juego.physics.arcade.gravity.y = 800;
juego.time.desiredFps = 30;
```

- Personaje y Enemigos: Se posicionan en el juego y se les asigna animaciones.
- Proyectiles: Se crean y posicionan proyectiles amigables y enemigos con propiedades físicas específicas.

## Funciones de Utilidad

- reiniciarBalaEnemiga: Reinicia la posición y velocidad de la bala enemiga.
- cambiarPielPersonaje: Cambia la textura del sprite del personaje dependiendo del contexto del juego.
- reiniciarPosicionBalaAmiga: Devuelve la bala amiga a su posición inicial.
- pausa y mPausa: Funciones para gestionar el estado de pausa del juego.

## Entrenamiento de la Red Neuronal

Se utilizan funciones para entrenar la red neuronal con datos de entrada simulados, y luego usar esa red para tomar decisiones basadas en el estado actual del juego.

```javascript
enRedNeuronal();
datosParaEntrenamiento(entrada);
datosParaEntrenamientoMovimiento(entrada);
datosParaEntrenamientoMovimiento3(entrada);
```

## Funciones principales

### resetearVariables()

Reinicia todas las variables y la posición de los objetos a sus estados iniciales.

```javascript
function resetearVariables() {
    protagonista.body.velocity.x = 0;
    protagonista.body.velocity.y = 0;
    bala.body.velocity.x = 0;
    bala.position.x = ancho - 100;
    balaDesplegada = false;
}
```

### saltar()

Permite al protagonista saltar dentro del juego al modificar su velocidad en el eje y.

```javascript
function saltar() {
    protagonista.body.velocity.y = -270;
}
```

### dezplazamientDerecha()

Controla el movimiento hacia la derecha del protagonista, asegurando que no se mueva más allá de un punto específico.

```javascript
const desplazamientoDerecha = () => {
    if (protagonista.body.position.x > 100)
        return;
    protagonista.body.position.x = 200;
    estadoDerecha = 1;
    estadoIzquierda = 0;
}
```

### desplazamientoIzquierda()

Mueve al protagonista hacia la izquierda y actualiza el estado del movimiento.

```javascript
const desplazamientoIzquierda = () => {
    protagonista.body.position.x = 0;
    estadoDerecha = 0;
    estadoIzquierda = 1;
    moviendose = true;
}
```

### update()

Es la función de actualización principal que se ejecuta en cada cuadro para actualizar los estados y posiciones de todos los objetos del juego.

### disparo()

Controla la lógica de disparo de una bala, estableciendo su velocidad y marcándola como desplegada.

```javascript
function disparo() {
    velocidadDisparo = -1 * velocidadAleatoria(100, 100);
    bala.body.velocity.y = 0;
    bala.body.velocity.x = velocidadDisparo;
    balaDesplegada = true;
}
```
### `colisionH()`

Gestiona las acciones a tomar cuando ocurre una colisión.

- **Funcionalidades:**
  - Cambia la apariencia del personaje.
  - Pausa el juego.
  - Restablece la posición del protagonista.

### `velocidadAleatoria(min, max)`

Genera un número aleatorio dentro de un rango especificado, utilizado para la velocidad del disparo.

- **Parámetros:**
  - `min`: Límite inferior del rango.
  - `max`: Límite superior del rango.

### `render()`

Función vacía para procesos de renderizado (usualmente más relevante en contextos donde se actualizan gráficos o se implementan efectos visuales).