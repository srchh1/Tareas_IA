# Documentación del Código de Videojuego con Phaser y Synaptic

Este documento describe el funcionamiento del código de un videojuego implementado con la librería Phaser para la creación del juego y Synaptic para la red neuronal que controla al personaje principal.

## Variables Globales

- `ancho`: Ancho del canvas del juego.
- `alto`: Alto del canvas del juego.
- `personaje`: Sprite del personaje principal.
- `escenario`: Fondo del juego.
- `proyectil`: Sprite del proyectil.
- `ventana`: Sprite del menú de pausa.
- `controles`: Controles del teclado.
- `teclas`: Teclas adicionales para control del personaje.
- `pausaTexto`: Texto para pausar el juego.
- `redNeural`: Red neuronal de Synaptic.
- `entrenador`: Entrenador para la red neuronal.
- `salidaNeural`: Salida de la red neuronal.
- `dataset`: Datos de entrenamiento para la red neuronal.
- `controlAuto`: Control para activar el modo automático.
- `entrenamientoCompleto`: Indica si el entrenamiento está completo.
- `PosX`: Posición X del personaje.
- `PosY`: Posición Y del personaje.
- `modoAutomático`: Indica si el modo automático está activado.
- `regresandoX`: Indica si el personaje está regresando en el eje X.
- `regresandoY`: Indica si el personaje está regresando en el eje Y.

## Funciones

### Función Principal

```javascript
var videojuego = new Phaser.Game(ancho, alto, Phaser.CANVAS, '', { preload: cargarRecursos, create: iniciar, update: actualizar, render: dibujar });
```

Inicializa el juego con Phaser, cargando recursos, creando objetos, actualizando el estado del juego y dibujando en pantalla.

## Cargar Recursos

```javascript
function cargarRecursos() {
    videojuego.load.image('escenario', 'assets/game/fondo1.jpg');
    videojuego.load.spritesheet('personaje', 'assets/sprites/dany.png', 43, 68);
    videojuego.load.image('ventana', 'assets/game/menu.png');
    videojuego.load.image('proyectil', 'assets/sprites/dan.png');
};
```
Carga las imágenes y sprites utilizados en el juego.

```javascript
function iniciar() {
    // Inicialización de física y FPS
    videojuego.physics.startSystem(Phaser.Physics.ARCADE);
    videojuego.time.desiredFps = 60;

    // Creación de sprites y habilitación de físicas
    escenario = videojuego.add.tileSprite(0, 0, ancho, alto, 'escenario');
    personaje = videojuego.add.sprite(ancho / 2, alto / 2, 'personaje');
    videojuego.physics.enable(personaje);
    personaje.body.collideWorldBounds = true;
    personaje.animations.add('mover', [8]);
    personaje.animations.play('mover', 7, true);

    proyectil = videojuego.add.sprite(700, 0, 'proyectil');
    videojuego.physics.enable(proyectil);
    proyectil.body.collideWorldBounds = true;
    proyectil.body.bounce.set(1);
    asignarVelocidadAleatoria();

    // Texto de pausa
    pausaTexto = videojuego.add.text(ancho - 70, 8, 'Pausas', { font: '20px Bodoni MT', fill: '#fff' });
    pausaTexto.inputEnabled = true;
    pausaTexto.events.onInputUp.add(togglePausa, self);
    videojuego.input.onDown.add(manejarPausa, self);

    // Controles del teclado
    controles = videojuego.input.keyboard.createCursorKeys();
    teclas = {
        up: videojuego.input.keyboard.addKey(Phaser.Keyboard.W),
        left: videojuego.input.keyboard.addKey(Phaser.Keyboard.A),
        down: videojuego.input.keyboard.addKey(Phaser.Keyboard.S),
        right: videojuego.input.keyboard.addKey(Phaser.Keyboard.D)
    };

    // Inicialización de la red neuronal
    redNeural = new synaptic.Architect.Perceptron(3, 6, 6, 6, 5);
    entrenador = new synaptic.Trainer(redNeural);
}:
```
Inicializa el juego, configurando físicas, FPS, sprites, controles y la red neuronal.

## Entrenar Red

```javascript
function entrenarRed() {
    entrenador.train(dataset, { rate: 0.001, iterations: 10000, shuffle: true, batchSize: 32 });
}
```
Entrena la red neuronal con los datos del dataset.

## Alternar Pausa

```javascript
function togglePausa() {
    videojuego.paused = true;
    ventana = videojuego.add.sprite(ancho / 2, alto / 2, 'ventana');
    ventana.anchor.setTo(0.5, 0.5);
}
```
Pausa el juego y muestra el menú de pausa.

## Manejar Pausa

```javascript
function manejarPausa(event) {
    if (videojuego.paused) {
        var x1 = ancho / 2 - 270 / 2, x2 = ancho / 2 + 270 / 2,
            y1 = alto / 2 - 180 / 2, y2 = alto / 2 + 180 / 2;

        var mouseX = event.x, mouseY = event.y;

        if (mouseX > x1 && mouseX < x2 && mouseY > y1 && mouseY < y2) {
            if (mouseY >= y1 && mouseY <= y1 + 90) {
                entrenamientoCompleto = false;
                dataset = [];
                controlAuto = false;
                modoAutomático = false;
            } else if (mouseY > y1 + 90 && mouseY <= y2) {
                if (!entrenamientoCompleto) {
                    entrenarRed();
                    entrenamientoCompleto = true;
                }
                controlAuto = true;
                modoAutomático = true;
            }
            ventana.destroy();
            reiniciarJuego();
            videojuego.paused = false;
        }
    }
}
```

Maneja los eventos de pausa y menú, entrenando la red si es necesario y reiniciando el juego.

## Reinciar Juego

```javascript
function reiniciarJuego() {
    personaje.x = ancho / 2;
    personaje.y = alto / 2;
    personaje.body.velocity.x = 100;
    personaje.body.velocity.y = 100;
    proyectil.x = 700;
    proyectil.y = 0;
    asignarVelocidadAleatoria();
}
```

Reinicia la posición del personaje y el proyectil.

## Asignar Velocidad Aleatoria

```javascript
function asignarVelocidadAleatoria() {
    var velocidad = 310;
    var ángulo = videojuego.rnd.angle();
    proyectil.body.velocity.set(Math.cos(ángulo) * velocidad, Math.sin(ángulo) * velocidad);
}
```

Asigna una velocidad aleatoria al proyectil.

## Actualizar

```javascript
function actualizar() {
    escenario.tilePosition.x -= 1;

    if (!modoAutomático) {
        personaje.body.velocity.x = 0;
        personaje.body.velocity.y = 0;

        if (controles.left.isDown || teclas.left.isDown) {
            personaje.body.velocity.x = -300;
        } else if (controles.right.isDown || teclas.right.isDown) {
            personaje.body.velocity.x = 300;
        }

        if (controles.up.isDown || teclas.up.isDown) {
            personaje.body.velocity.y = -300;
        } else if (controles.down.isDown || teclas.down.isDown) {
            personaje.body.velocity.y = 300;
        }
    }

    videojuego.physics.arcade.collide(proyectil, personaje, colisiónPersonaje, null, this);

    var dx = proyectil.x - personaje.x;
    var dy = proyectil.y - personaje.y;
    var distancia = Math.sqrt(dx * dx + dy * dy);

    if (controlAuto && evaluarMovimiento([dx, dy, distancia, PosX, PosY])) {
        if (distancia <= 1000) {
            if (evaluarVertical([dx, dy, distancia, PosX, PosY]) && !regresandoY) {
                personaje.body.velocity.y -= 35;
            } else if (!evaluarVertical([dx, dy, distancia, PosX, PosY]) && !regresandoY && distancia <= 95) {
                personaje.body.velocity.y += 35;
            }

            if (evaluarHorizontal([dx, dy, distancia, PosX, PosY]) && !regresandoX) {
                personaje.body.velocity.x -= 35;
            } else if (!evaluarHorizontal([dx, dy, distancia, PosX, PosY]) && !regresandoX && distancia <= 95) {
                personaje.body.velocity.x += 35;
                }
                if (personaje.x > 300) {
                    personaje.body.velocity.x = -350;
                    regresandoX = true;
                } else if (personaje.x < 100) {
                    personaje.body.velocity.x = 350;
                    regresandoX = true;
                } else if (regresandoX && personaje.x > 150 && personaje.x < 250) {
                    personaje.body.velocity.x = 0;
                    regresandoX = false;
                }
        
                if (personaje.y > 300) {
                    personaje.body.velocity
```
